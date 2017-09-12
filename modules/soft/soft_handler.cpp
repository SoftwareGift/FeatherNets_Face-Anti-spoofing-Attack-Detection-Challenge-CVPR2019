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
    SyncMeta () : _done (false) {}
    void signal_done (XCamReturn err);
    void wakeup ();
    XCamReturn signal_wait_ret ();

private:
    Mutex       _mutex;
    Cond        _cond;
    bool        _done;
    XCamReturn  _error;
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

SoftHandler::SoftHandler (const char* name)
    : ImageHandler (name)
    , _need_configure (true)
    , _wip_buf_count (0)
{
    set_allocator (new SoftVideoBufAllocator);
    char thrds_name[XCAM_MAX_STR_SIZE];
    snprintf (thrds_name, XCAM_MAX_STR_SIZE, "t-pool-%s", XCAM_STR(name));
    _threads = new ThreadPool (thrds_name);
}

SoftHandler::~SoftHandler ()
{
}

bool
SoftHandler::set_threads (uint32_t num)
{
    return _threads->set_threads (num, num);
}

XCamReturn
SoftHandler::configure_resource (const SmartPtr<ImageHandler::Parameters> &params)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (!_need_configure)
        return ret;

    if (!params->out_buf.ptr ()) {
        ret = reserve_buffers (params->out_buf->get_video_info (), DEFAULT_SOFT_BUF_COUNT);
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "soft_hander(%s) configure resource failed in reserving buffers", XCAM_STR (get_name ()));
        _need_configure = false;
    }

    if (_threads.ptr ()) {
        ret = _threads->start ();
        XCAM_FAIL_RETURN (
            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
            "soft_hander(%s) configure resource failed when starting threads", XCAM_STR (get_name ()));
    }

    return ret;
}

XCamReturn
SoftHandler::execute_buffer (SmartPtr<ImageHandler::Parameters> &params, bool sync)
{
    SmartPtr<SyncMeta> sync_meta;
    XCamReturn ret = configure_resource (params);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "soft_hander(%s) configure resource failed", XCAM_STR (get_name ()));

    if (sync) {
        XCAM_ASSERT (!params->find_meta<SyncMeta> ().ptr ());
        sync_meta = new SyncMeta ();
        XCAM_ASSERT (sync_meta.ptr ());
        params->add_meta (sync_meta);
    }

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

    XCAM_FAIL_RETURN (
        WARNING, ret >= XCAM_RETURN_NO_ERROR, XCAM_RETURN_ERROR_PARAM,
        "soft_hander(%s) execute buffer failed in working", XCAM_STR (get_name ()));
    ++_wip_buf_count;

    if (sync) {
        XCAM_ASSERT (sync_meta.ptr ());
        _cur_sync = sync_meta;
        ret = sync_meta->signal_wait_ret ();
        _cur_sync.release ();
    }

    return ret;
}

XCamReturn
SoftHandler::finish ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    if (_cur_sync.ptr ()) {
        ret = _cur_sync->signal_wait_ret ();
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
    if (_cur_sync.ptr ()) {
        _cur_sync->wakeup ();
        _cur_sync.release ();
    }
    _params.clear ();
    return ImageHandler::terminate ();
}

XCamReturn
SoftHandler::last_worker_done (XCamReturn err)
{
    SmartPtr<ImageHandler::Parameters> params = _params.pop (0);
    SmartPtr<SyncMeta> sync_meta = params->find_meta<SyncMeta> ();
    if (sync_meta.ptr ())
        sync_meta->signal_done (err);

    --_wip_buf_count;
    XCAM_ASSERT (params.ptr ());
    return execute_status_check (params, err);
}

}

