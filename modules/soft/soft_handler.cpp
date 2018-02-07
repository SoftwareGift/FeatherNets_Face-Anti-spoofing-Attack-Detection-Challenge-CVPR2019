/*
 * soft_handler.cpp - soft image handler implementation
 *
 *  Copyright (c) 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "soft_handler.h"
#include "soft_video_buf_allocator.h"
#include "thread_pool.h"
#include "soft_worker.h"

#define DEFAULT_SOFT_BUF_COUNT 4

namespace XCam {

class SyncMeta
    : public MetaBase
{
public:
    SyncMeta ()
        : _done (false)
        , _error (XCAM_RETURN_NO_ERROR) {}
    void signal_done (XCamReturn err);
    void wakeup ();
    XCamReturn signal_wait_ret ();
    bool is_error () const;

private:
    mutable Mutex   _mutex;
    Cond            _cond;
    bool            _done;
    XCamReturn      _error;
};

void
SyncMeta::signal_done (XCamReturn err)
{
    SmartLock locker (_mutex);
    _done = true;
    _error = err;
    _cond.broadcast ();
}

void
SyncMeta::wakeup ()
{
    SmartLock locker (_mutex);
    _error = XCAM_RETURN_ERROR_UNKNOWN;
    _cond.broadcast ();
}

XCamReturn
SyncMeta::signal_wait_ret ()
{
    SmartLock locker (_mutex);
    if (_done)
        return _error;
    _cond.wait (_mutex);
    return _error;
}

bool
SyncMeta::is_error () const
{
    SmartLock locker (_mutex);
    return !xcam_ret_is_ok (_error);
}

SoftHandler::SoftHandler (const char* name)
    : ImageHandler (name)
    , _need_configure (true)
    , _enable_allocator (true)
    , _wip_buf_count (0)
{
}

SoftHandler::~SoftHandler ()
{
}

bool
SoftHandler::set_threads (const SmartPtr<ThreadPool> &pool)
{
    _threads = pool;
    return true;
}

bool
SoftHandler::set_out_video_info (const VideoBufferInfo &info)
{
    XCAM_ASSERT (info.width && info.height && info.format);
    _out_video_info = info;
    return true;
}

bool
SoftHandler::enable_allocator (bool enable)
{
    _enable_allocator = enable;
    return true;
}

XCamReturn
SoftHandler::confirm_configured ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_need_configure);
    if (_enable_allocator) {
        XCAM_FAIL_RETURN (
            ERROR, _out_video_info.is_valid (), XCAM_RETURN_ERROR_PARAM,
            "soft_hander(%s) configure resource failed before reserver buffer since out video info was not set",
            XCAM_STR (get_name ()));

        set_allocator (new SoftVideoBufAllocator);
        ret = reserve_buffers (_out_video_info, DEFAULT_SOFT_BUF_COUNT);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "soft_hander(%s) configure resource failed in reserving buffers", XCAM_STR (get_name ()));
    }

    if (_threads.ptr () && !_threads->is_running ()) {
        ret = _threads->start ();
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "soft_hander(%s) configure resource failed when starting threads", XCAM_STR (get_name ()));
    }
    _need_configure = false;

    return ret;
}

XCamReturn
SoftHandler::execute_buffer (const SmartPtr<ImageHandler::Parameters> &param, bool sync)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        ERROR, param.ptr (), XCAM_RETURN_ERROR_PARAM,
        "soft_hander(%s) execute buffer failed, params is null",
        XCAM_STR (get_name ()));

    if (_need_configure) {
        ret = configure_resource (param);
        XCAM_FAIL_RETURN (
            WARNING, xcam_ret_is_ok (ret), ret,
            "soft_hander(%s) configure resource failed", XCAM_STR (get_name ()));

        ret = confirm_configured ();
        XCAM_FAIL_RETURN (
            WARNING, xcam_ret_is_ok (ret), ret,
            "soft_hander(%s) confirm configure failed", XCAM_STR (get_name ()));
    }

    if (!param->out_buf.ptr () && _enable_allocator) {
        param->out_buf = get_free_buf ();
        XCAM_FAIL_RETURN (
            ERROR, param->out_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
            "soft_hander:%s execute buffer failed, output buffer failed in allocation.",
            XCAM_STR (get_name ()));
    }

    XCAM_ASSERT (!param->find_meta<SyncMeta> ().ptr ());
    SmartPtr<SyncMeta> sync_meta = new SyncMeta ();
    XCAM_ASSERT (sync_meta.ptr ());
    param->add_meta (sync_meta);

#if 0
    SmartPtr<SoftWorker> worker = get_first_worker ().dynamic_cast_ptr<SoftWorker> ();
    XCAM_FAIL_RETURN (
        WARNING, worker.ptr (), XCAM_RETURN_ERROR_PARAM,
        "No worder set to soft_hander(%s)", XCAM_STR (get_name ()));

    SmartPtr<Worker::Arguments> args = get_first_worker_args (worker, params);
    XCAM_FAIL_RETURN (
        WARNING, args.ptr (), XCAM_RETURN_ERROR_PARAM,
        "soft_hander(%s) get first worker(%s) args failed",
        XCAM_STR (get_name ()), XCAM_STR (worker->get_name ()));

    _params.push (params);
    ret = worker->work (args);
#else
    _params.push (param);
    ret = start_work (param);
#endif

    if (!xcam_ret_is_ok (ret)) {
        _params.erase (param);
        XCAM_LOG_WARNING ("soft_hander(%s) execute buffer failed in starting workers", XCAM_STR (get_name ()));
        return ret;
    }

    ++_wip_buf_count;
    _cur_sync = sync_meta;

    if (sync) {
        XCAM_ASSERT (sync_meta.ptr ());
        ret = sync_meta->signal_wait_ret ();
        _cur_sync.release ();
    }

    return ret;
}

XCamReturn
SoftHandler::finish ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<SyncMeta> sync = _cur_sync;
    if (sync.ptr ()) {
        ret = sync->signal_wait_ret ();
    }
    XCAM_ASSERT (_params.is_empty ());
    //wait for _wip_buf_count = 0
    //if (ret == XCAM_RETURN_NO_ERROR)
    //    XCAM_ASSERT (_wip_buf_count == 0);

    return ret;
}

XCamReturn
SoftHandler::terminate ()
{
    SmartPtr<SyncMeta> sync = _cur_sync;
    if (sync.ptr ()) {
        sync->wakeup ();
        sync.release ();
    }
    _params.clear ();
    return ImageHandler::terminate ();
}

void
SoftHandler::work_well_done (const SmartPtr<ImageHandler::Parameters> &param, XCamReturn err)
{
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (xcam_ret_is_ok (err));

    if (!xcam_ret_is_ok (err)) {
        XCAM_LOG_WARNING ("soft_hander(%s) work_well_done but errno(%d) is not ok", XCAM_STR (get_name ()), (int)err);
        //continue work
    }

    if (!_params.erase (param)) {
        XCAM_LOG_ERROR(
            "soft_hander(%s) last_work_done param already removed, who removed it?", XCAM_STR (get_name ()));
        return;
    }

    XCAM_LOG_DEBUG ("soft_hander(%s) work well done", XCAM_STR (get_name ()));

    param_ended (param, err);
}

void
SoftHandler::work_broken (const SmartPtr<ImageHandler::Parameters> &param, XCamReturn err)
{
    XCAM_ASSERT (param.ptr ());
    XCAM_ASSERT (!xcam_ret_is_ok (err));

    if (xcam_ret_is_ok (err)) {
        XCAM_LOG_WARNING ("soft_hander(%s) work_broken but the errno(%d) is ok", XCAM_STR (get_name ()), (int)err);
        //continue work
    }

    if (!_params.erase (param)) {
        //already removed by other handlers
        return;
    }
    XCAM_LOG_WARNING ("soft_hander(%s) work broken", XCAM_STR (get_name ()));

    param_ended (param, err);
}

void
SoftHandler::param_ended (SmartPtr<ImageHandler::Parameters> param, XCamReturn err)
{
    XCAM_ASSERT (param.ptr ());

    SmartPtr<SyncMeta> sync_meta = param->find_meta<SyncMeta> ();
    XCAM_ASSERT (sync_meta.ptr ());
    sync_meta->signal_done (err);
    --_wip_buf_count;
    execute_status_check (param, err);
}

bool
SoftHandler::check_work_continue (const SmartPtr<ImageHandler::Parameters> &param, XCamReturn err)
{
    if (!xcam_ret_is_ok (err)) {
        work_broken (param, err);
        return false;
    }

    if (is_param_error (param)) {
        XCAM_LOG_WARNING (
            "soft_handler(%s) check_work_continue found param broken", XCAM_STR(get_name ()));
        return false;
    }
    return true;
}

bool
SoftHandler::is_param_error (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr ());
    SmartPtr<SyncMeta> meta = param->find_meta<SyncMeta> ();
    if (!meta.ptr ()) { // return ok if param not set
        XCAM_ASSERT (meta.ptr ());
        return false;
    }

    return meta->is_error ();
}

}

