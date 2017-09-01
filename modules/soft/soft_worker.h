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

#include "xcam_utils.h"
#include "worker.h"

namespace XCam {

class ThreadPool;

//multi-thread worker
class SoftWorker
    : public Worker
{
    friend class WorkItem;

public:
    struct WorkSize {
        uint32_t x, y, z;
        WorkSize (uint32_t x0 = 1, uint32_t y0 = 1, uint32_t z0 = 1) : x(x0), y(y0), z(z0) {}
    };

public:
    explicit SoftWorker (const char *name);
    virtual ~SoftWorker ();
    bool set_threads (const SmartPtr<ThreadPool> &threads);
    bool set_work_size (const WorkSize &size);
    const WorkSize &get_work_size () const {
        return _work_size;
    }

    // derived from Worker
    virtual XCamReturn work (const SmartPtr<Arguments> &args);

private:
    virtual XCamReturn work_impl (const SmartPtr<Arguments> &args, const WorkSize &item) = 0;
    void all_items_done (const SmartPtr<Arguments> &args, XCamReturn error);

    XCAM_DEAD_COPY (SoftWorker);

private:
    SmartPtr<ThreadPool>    _threads;
    WorkSize                _work_size;
};

}
#endif //XCAM_SOFT_WORKER_H
