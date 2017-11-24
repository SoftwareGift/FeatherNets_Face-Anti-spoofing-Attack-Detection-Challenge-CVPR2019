/*
 * soft_worker.h - soft worker class
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

#ifndef XCAM_SOFT_WORKER_H
#define XCAM_SOFT_WORKER_H

#include <xcam_std.h>
#include <worker.h>

#define SOFT_MAX_DIM 3

namespace XCam {

class ThreadPool;

struct WorkRange {
    uint32_t pos[SOFT_MAX_DIM];
    uint32_t pos_len[SOFT_MAX_DIM];

    WorkRange () {
        xcam_mem_clear (pos);
        xcam_mem_clear (pos_len);
    }
};

struct WorkSize {
    uint32_t value[SOFT_MAX_DIM];
    WorkSize (uint32_t x = 1, uint32_t y = 1, uint32_t z = 1) {
        value[0] = x;
        value[1] = y;
        value[2] = z;
    }
};

//multi-thread worker
class SoftWorker
    : public Worker
{
    friend class WorkItem;

public:
    explicit SoftWorker (const char *name, const SmartPtr<Callback> &cb = NULL);
    virtual ~SoftWorker ();

    bool set_work_uint (uint32_t x, uint32_t y, uint32_t z = 1);
    const WorkSize &get_work_uint () const {
        return _work_unit;
    }

    bool set_threads (const SmartPtr<ThreadPool> &threads);
    bool set_global_size (const WorkSize &size);
    const WorkSize &get_global_size () const {
        return _global;
    }
    bool set_local_size (const WorkSize &size);
    const WorkSize &get_local_size () const {
        return _local;
    }

    // derived from Worker
    virtual XCamReturn work (const SmartPtr<Arguments> &args);
    virtual XCamReturn stop ();

private:
    //new virtual functions
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
    virtual WorkRange get_range (const WorkSize &item);
    virtual XCamReturn work_unit (const SmartPtr<Arguments> &args, const WorkSize &unit);

    XCamReturn work_impl (const SmartPtr<Arguments> &args, const WorkSize &item);
    void all_items_done (const SmartPtr<Arguments> &args, XCamReturn error);

    XCAM_DEAD_COPY (SoftWorker);

private:
    SmartPtr<ThreadPool>    _threads;
    WorkSize                _global;
    WorkSize                _local;
    WorkSize                _work_unit;
};

}
#endif //XCAM_SOFT_WORKER_H
