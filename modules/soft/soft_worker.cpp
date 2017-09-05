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
        const SoftWorker::WorkSize &item,
        SmartPtr<ItemSynch> &sync)
        : _worker (worker)
        , _args (args)
        , _item (item)
        , _sync (sync)
        , _error (XCAM_RETURN_NO_ERROR)
    {
    }
    virtual XCamReturn run ();
    virtual void done (XCamReturn err);


private:
    SmartPtr<SoftWorker>         _worker;
    SmartPtr<Worker::Arguments>  _args;
    SoftWorker::WorkSize         _item;
    SmartPtr<ItemSynch>          _sync;
    XCamReturn                   _error;
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

SoftWorker::SoftWorker (const char *name)
    : Worker (name)
{
}

SoftWorker::~SoftWorker ()
{
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
SoftWorker::set_work_size (const WorkSize &size)
{
    _work_size = size;
    return true;
}

XCamReturn
SoftWorker::work (const SmartPtr<Worker::Arguments> &args)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (_work_size.x == 1 && _work_size.y == 1 && _work_size.z == 1) {
        XCAM_ASSERT (_work_size.x == 1 && _work_size.y == 1 && _work_size.z == 1);
        ret = work_impl (args, _work_size);
        return status_check (args, ret);
    }

    uint32_t max_items = _work_size.x * _work_size.y * _work_size.z;
    if (!_threads.ptr ()) {
        char thr_name [XCAM_MAX_STR_SIZE];
        snprintf (thr_name, XCAM_MAX_STR_SIZE, "%s-thread-pool", XCAM_STR(get_name ()));
        _threads = new ThreadPool (thr_name);
        XCAM_ASSERT (_threads.ptr ());
        _threads->set_threads (max_items, max_items);
        ret = _threads->start ();
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "SoftWorker(%s) work failed when starting threads", XCAM_STR(get_name()));
    }

    SmartPtr<ItemSynch> sync = new ItemSynch (_work_size.x * _work_size.y * _work_size.z);
    for (uint32_t z = 0; z < _work_size.z; ++z)
        for (uint32_t y = 0; y < _work_size.y; ++y)
            for (uint32_t x = 0; x < _work_size.x; ++x)
            {
                SmartPtr<WorkItem> item = new WorkItem (this, args, WorkSize(x, y, z), sync);
                ret = _threads->queue (item);
                XCAM_FAIL_RETURN (
                    ERROR, xcam_ret_is_ok (ret), ret,
                    "SoftWorker(%s) queue work item(x:%d y: %d z:%d) failed",
                    _work_size.x, _work_size.y, _work_size.z);
            }

    return XCAM_RETURN_NO_ERROR;
}

void
SoftWorker::all_items_done (const SmartPtr<Arguments> &args, XCamReturn error)
{
    status_check (args, error);
}

};
