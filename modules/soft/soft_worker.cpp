/*
 * soft_worker.cpp - soft worker implementation
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

#include "soft_worker.h"
#include "thread_pool.h"
#include "xcam_mutex.h"

namespace XCam {

class ItemSynch {
private:
    mutable std::atomic<uint32_t>  _remain_items;
    Mutex                          _mutex;
    XCamReturn                     _error;

public:
    ItemSynch (uint32_t items)
        : _remain_items(items), _error (XCAM_RETURN_NO_ERROR)
    {}
    void update_error (XCamReturn err) {
        SmartLock locker(_mutex);
        _error = err;
    }
    XCamReturn get_error () {
        SmartLock locker(_mutex);
        return _error;
    }
    uint32_t dec() {
        return --_remain_items;
    }

private:
    XCAM_DEAD_COPY (ItemSynch);
};

class WorkItem
    : public ThreadPool::UserData
{
public:
    WorkItem (
        const SmartPtr<SoftWorker> &worker,
        const SmartPtr<Worker::Arguments> &args,
        const WorkSize &item,
        SmartPtr<ItemSynch> &sync)
        : _worker (worker)
        , _args (args)
        , _item (item)
        , _sync (sync)
    {
    }
    virtual XCamReturn run ();
    virtual void done (XCamReturn err);


private:
    SmartPtr<SoftWorker>         _worker;
    SmartPtr<Worker::Arguments>  _args;
    WorkSize                     _item;
    SmartPtr<ItemSynch>          _sync;
};

XCamReturn
WorkItem::run ()
{
    XCamReturn ret = _sync->get_error();
    if (!xcam_ret_is_ok (ret))
        return ret;

    ret = _worker->work_impl (_args, _item);
    if (!xcam_ret_is_ok (ret))
        _sync->update_error (ret);

    return ret;
}

void
WorkItem::done (XCamReturn err)
{
    if (_sync->dec () == 0) {
        XCamReturn ret = _sync->get_error ();
        if (xcam_ret_is_ok (ret))
            ret = err;
        _worker->all_items_done (_args, ret);
    }
}

SoftWorker::SoftWorker (const char *name, const SmartPtr<Callback> &cb)
    : Worker (name, cb)
    , _global (1, 1, 1)
    , _local (1, 1, 1)
    , _work_unit (1, 1, 1)
{
}

SoftWorker::~SoftWorker ()
{
}

bool
SoftWorker::set_work_uint (uint32_t x, uint32_t y, uint32_t z)
{
    XCAM_FAIL_RETURN (
        ERROR, x && y && z, false,
        "SoftWorker(%s) set work unit failed(x:%d, y:%d, z:%d)",
        XCAM_STR (get_name ()), x, y, z);
    _work_unit.value[0] = x;
    _work_unit.value[1] = y;
    _work_unit.value[2] = z;
    return true;
}

bool
SoftWorker::set_threads (const SmartPtr<ThreadPool> &threads)
{
    XCAM_FAIL_RETURN (
        ERROR, !_threads.ptr (), false,
        "SoftWorker(%s) set threads failed, it's already set before.", XCAM_STR (get_name ()));
    _threads = threads;
    return true;
}

bool
SoftWorker::set_global_size (const WorkSize &size)
{
    XCAM_FAIL_RETURN (
        ERROR, size.value[0] && size.value[1] && size.value[2], false,
        "SoftWorker(%s) set global size(x:%d, y:%d, z:%d) failed.",
        XCAM_STR (get_name ()), size.value[0], size.value[1], size.value[2]);

    _global = size;
    return true;
}

bool
SoftWorker::set_local_size (const WorkSize &size)
{
    XCAM_FAIL_RETURN (
        ERROR, size.value[0] && size.value[1] && size.value[2], false,
        "SoftWorker(%s) set local size(x:%d, y:%d, z:%d) failed.",
        XCAM_STR (get_name ()), size.value[0], size.value[1], size.value[2]);

    _local = size;
    return true;
}

XCamReturn
SoftWorker::stop ()
{
    _threads->stop ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SoftWorker::work (const SmartPtr<Worker::Arguments> &args)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_local.value[0] * _local.value[1] * _local.value[2]);
    XCAM_ASSERT (_global.value[0] * _global.value[1] * _global.value[2]);

    WorkSize items;
    uint32_t max_items = 1;

    for (uint32_t i = 0; i < SOFT_MAX_DIM; ++i) {
        items.value[i] = xcam_ceil (_global.value[i],  _local.value[i]) / _local.value[i];
        max_items *= items.value[i];
    }

    XCAM_FAIL_RETURN (
        ERROR, max_items, XCAM_RETURN_ERROR_PARAM,
        "SoftWorker(%s) max item is zero. work failed.", XCAM_STR (get_name ()));

    if (max_items == 1) {
        ret = work_impl (args, WorkSize(0, 0, 0));
        status_check (args, ret);
        return ret;
    }

    if (!_threads.ptr ()) {
        char thr_name [XCAM_MAX_STR_SIZE];
        snprintf (thr_name, XCAM_MAX_STR_SIZE, "%s-thrs", XCAM_STR(get_name ()));

        SmartPtr<ThreadPool> threads = new ThreadPool (thr_name);
        XCAM_ASSERT (threads.ptr ());
        _threads = threads;
        _threads->set_threads (max_items, max_items + 1); //extra thread to process all_items_done
        ret = _threads->start ();
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "SoftWorker(%s) work failed when starting threads", XCAM_STR(get_name()));
    }

    SmartPtr<ItemSynch> sync = new ItemSynch (max_items);
    for (uint32_t z = 0; z < items.value[2]; ++z)
        for (uint32_t y = 0; y < items.value[1]; ++y)
            for (uint32_t x = 0; x < items.value[0]; ++x)
            {
                SmartPtr<WorkItem> item = new WorkItem (this, args, WorkSize(x, y, z), sync);
                ret = _threads->queue (item);
                if (!xcam_ret_is_ok (ret)) {
                    //consider half queued but half failed
                    sync->update_error (ret);
                    //status_check (args, ret); // need it here?
                    XCAM_LOG_ERROR (
                        "SoftWorker(%s) queue work item(x:%d y: %d z:%d) failed",
                        XCAM_STR(get_name()), x, y, z);
                    return ret;
                }
            }

    return XCAM_RETURN_NO_ERROR;
}

void
SoftWorker::all_items_done (const SmartPtr<Arguments> &args, XCamReturn error)
{
    status_check (args, error);
}

WorkRange
SoftWorker::get_range (const WorkSize &item)
{
    WorkRange range;
    for (uint32_t i = 0; i < SOFT_MAX_DIM; ++i) {
        range.pos[i] = item.value[i] * _local.value[i];
        XCAM_ASSERT (range.pos[i] < _global.value[i]);
        if (range.pos[i] + _local.value[i] > _global.value[i])
            range.pos_len[i] = _global.value[i] - range.pos[i];
        else
            range.pos_len[i] = _local.value[i];
    }
    return range;
}

XCamReturn
SoftWorker::work_impl (const SmartPtr<Arguments> &args, const WorkSize &item)
{
    WorkRange range = get_range (item);
    return work_range (args, range);
}

XCamReturn
SoftWorker::work_range (const SmartPtr<Arguments> &args, const WorkRange &range)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    WorkSize unit;
    memcpy(unit.value, range.pos, sizeof (unit.value));

    for (unit.value[2] = range.pos[2]; unit.value[2] < range.pos[2] + range.pos_len[2]; ++unit.value[2])
        for (unit.value[1] = range.pos[1]; unit.value[1] < range.pos[1] + range.pos_len[1]; ++unit.value[1])
            for (unit.value[0] = range.pos[0]; unit.value[0] < range.pos[0] + range.pos_len[0]; ++unit.value[0]) {
                ret = work_unit (args, unit);
                XCAM_FAIL_RETURN (
                    ERROR, xcam_ret_is_ok (ret), ret,
                    "SoftWorker(%s) work on pixel(x:%d y: %d z:%d) failed",
                    get_name (), unit.value[0], unit.value[1], unit.value[2]);
            }

    return ret;
}

XCamReturn
SoftWorker::work_unit (const SmartPtr<Arguments> &, const WorkSize &)
{
    XCAM_LOG_ERROR ("SoftWorker(%s) work_pixel was not derived. check code");
    return XCAM_RETURN_ERROR_PARAM;
}

};
