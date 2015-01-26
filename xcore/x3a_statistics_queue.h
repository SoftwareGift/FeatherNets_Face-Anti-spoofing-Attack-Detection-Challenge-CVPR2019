/*
 * x3a_statistics_queue.h - statistics queue
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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

#ifndef XCAM_3A_STATISTIC_QUEUE_H
#define XCAM_3A_STATISTIC_QUEUE_H

#include "xcam_utils.h"
#include "x3a_event.h"
#include "xcam_mutex.h"
#include "xcam_utils.h"
#include "smartptr.h"
#include <queue>
#include <linux/atomisp.h>

namespace XCam {

class X3aStatisticsQueue;

class X3aIspStatistics
    : public X3aEvent
{
    friend class X3aStatisticsQueue;
protected:
    explicit X3aIspStatistics (
        X3aStatisticsQueue* pool,
        struct atomisp_3a_statistics *stats,
        uint64_t timestamp =  XCam::InvalidTimestamp
    )
        : X3aEvent (X3aEvent::TYPE_ISP_STATISTICS, timestamp)
        , _3a_pool (pool)
        , _3a_stats (stats)
        , _timestamp (timestamp)
    {}
public:
    virtual ~X3aIspStatistics ();
    const struct atomisp_3a_statistics *get_3a_stats () const {
        return _3a_stats;
    }
    struct atomisp_3a_statistics *get_3a_stats () {
        return _3a_stats;
    }

    int64_t get_timestamp () const {
        return _timestamp;
    }
    void set_timestamp (int64_t timestamp) {
        _timestamp = timestamp;
    }

private:
    XCAM_DEAD_COPY (X3aIspStatistics);

private:
    X3aStatisticsQueue           *_3a_pool;
    struct atomisp_3a_statistics *_3a_stats;
    int64_t                       _timestamp;
};

class X3aStatisticsQueue {
    friend class X3aIspStatistics;
    typedef std::queue<struct atomisp_3a_statistics *> StatsList;
public:
    explicit X3aStatisticsQueue ();
    ~X3aStatisticsQueue();

    void set_grid_info (struct atomisp_grid_info &info) {
        _grid_info = info;
    }

    SmartPtr<X3aIspStatistics> acquire_stats ();
    void wakeup ();

private:
    void release_stats (struct atomisp_3a_statistics *_3a_stats);
    bool pre_alloc_stats ();
    void clear ();

    XCAM_DEAD_COPY (X3aStatisticsQueue);

private:
    StatsList   _3a_stats_list;
    uint32_t    _list_size;
    bool        _list_allocated;
    XCam::Mutex _list_mutex;
    XCam::Cond  _list_release;
    struct atomisp_grid_info  _grid_info;
};

};

#endif //XCAM_3A_STATISTIC_QUEUE_H
