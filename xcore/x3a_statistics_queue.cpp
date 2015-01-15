/*
 * x3a_statistics_queue.c - statistics queue
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

#include "x3a_statistics_queue.h"
#include <linux/videodev2.h>
#include <linux/atomisp.h>

namespace XCam {

X3aIspStatistics::~X3aIspStatistics ()
{
    if (_3a_pool && _3a_stats)
        _3a_pool->release_stats(_3a_stats);
}

X3aStatisticsQueue::X3aStatisticsQueue()
    : _list_size(6)
    , _list_allocated(false)
{
    xcam_mem_clear (&_grid_info);
}

X3aStatisticsQueue::~X3aStatisticsQueue()
{
    clear ();
}

bool
X3aStatisticsQueue::pre_alloc_stats ()
{
    XCAM_ASSERT (_grid_info.width && _grid_info.height);
    XCAM_ASSERT (_grid_info.aligned_width && _grid_info.aligned_height);
    XCAM_ASSERT (_list_size);
    XCAM_ASSERT (_3a_stats_list.empty());

    uint32_t grid_size = _grid_info.aligned_width * _grid_info.aligned_height;
    //uint32_t grid_size = _grid_info.width * _grid_info.height;

    for (uint32_t i = 0; i < _list_size; ++i) {
        struct atomisp_3a_statistics *stats = xcam_malloc0_type (struct atomisp_3a_statistics);
        stats->data = (struct atomisp_3a_output*)xcam_malloc0 (grid_size * sizeof(*stats->data));
        XCAM_ASSERT (stats && stats->data);
        if (!stats || !stats->data)
            return false;
        stats->grid_info = _grid_info;
        _3a_stats_list.push (stats);
    }

    _list_allocated = true;
    return true;
}

SmartPtr<X3aIspStatistics>
X3aStatisticsQueue::acquire_stats ()
{
    struct atomisp_3a_statistics *stats;

    SmartLock locker(_list_mutex);

    if (!_list_allocated && !pre_alloc_stats ()) {
        XCAM_LOG_ERROR ("prealloc 3a stats failed");
        return NULL;
    }

    if (_3a_stats_list.empty())
        _list_release.wait (_list_mutex);

    if (_3a_stats_list.empty()) {
        XCAM_LOG_WARNING ("3a stats list waked up but still empty, check why");
        return NULL;
    }

    stats = _3a_stats_list.front();
    XCAM_ASSERT (stats);
    _3a_stats_list.pop ();
    return new X3aIspStatistics(this, stats);

}

void
X3aStatisticsQueue::wakeup ()
{
    XCAM_LOG_INFO ("wakeup all stats in acquiring");
    _list_release.broadcast ();
}

void
X3aStatisticsQueue::release_stats (struct atomisp_3a_statistics *_3a_stats)
{
    SmartLock locker(_list_mutex);
    _3a_stats_list.push (_3a_stats);
    XCAM_ASSERT (_3a_stats_list.size() <= _list_size);
    _list_release.signal ();
}

void
X3aStatisticsQueue::clear ()
{
    XCAM_ASSERT (!_list_allocated || _3a_stats_list.size() == _list_size);
    while (!_3a_stats_list.empty()) {
        struct atomisp_3a_statistics *stats = _3a_stats_list.front ();
        _3a_stats_list.pop ();
        xcam_free (stats->data);
        xcam_free (stats);
    }
}

};
