/*
 * x3a_stats_pool.cpp -  3a stats pool
 *
 *  Copyright (c) 2015 Intel Corporation
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

#include "x3a_stats_pool.h"

#define XCAM_3A_STATS_DEFAULT_BIT_DEPTH 8

namespace XCam {

X3aStatsData::X3aStatsData (XCam3AStats *data)
    : _data (data)
{
    XCAM_ASSERT (_data);
}

X3aStatsData::~X3aStatsData ()
{
    if (_data)
        xcam_free (_data);
}

uint8_t *
X3aStatsData::map ()
{
    return (uint8_t*)(intptr_t)(_data);
}

bool
X3aStatsData::unmap ()
{
    return true;
}

X3aStats::X3aStats (const SmartPtr<X3aStatsData> &data)
    : BufferProxy (SmartPtr<BufferData>(data))
{
}


XCam3AStats *
X3aStats::get_stats ()
{
    SmartPtr<BufferData> data = get_buffer_data ();
    SmartPtr<X3aStatsData> stats = data.dynamic_cast_ptr<X3aStatsData> ();

    XCAM_FAIL_RETURN(
        WARNING,
        stats.ptr(),
        NULL,
        "X3aStats get_stats failed with NULL");
    return stats->get_stats ();
}

X3aStatsPool::X3aStatsPool ()
    : _bit_depth (XCAM_3A_STATS_DEFAULT_BIT_DEPTH)
{
}

void
X3aStatsPool::set_stats_info (const XCam3AStatsInfo &info)
{
    _stats_info = info;
}

bool
X3aStatsPool::fixate_video_info (VideoBufferInfo &info)
{
    const uint32_t grid = 16;

    _stats_info.aligned_width = (info.width + grid - 1) / grid;
    _stats_info.aligned_height = (info.height + grid - 1) / grid;

    _stats_info.width = info.width / grid;
    _stats_info.height = info.height / grid;
    _stats_info.grid_pixel_size = grid;
    _stats_info.bit_depth = _bit_depth;
    _stats_info.histogram_bins = (1 << _bit_depth);
    return true;
}

SmartPtr<BufferData>
X3aStatsPool::allocate_data (const VideoBufferInfo &buffer_info)
{
    XCAM_UNUSED (buffer_info);

    XCam3AStats *stats = NULL;
    stats =
        (XCam3AStats *) xcam_malloc0 (
            sizeof (XCam3AStats) +
            sizeof (XCamHistogram) * _stats_info.histogram_bins +
            sizeof (uint32_t) * _stats_info.histogram_bins +
            sizeof (XCamGridStat) * _stats_info.aligned_width * _stats_info.aligned_height);
    XCAM_ASSERT (stats);
    stats->info = _stats_info;
    stats->hist_rgb = (XCamHistogram *) (stats->stats +
                                         _stats_info.aligned_width * _stats_info.aligned_height);
    stats->hist_y = (uint32_t *) (stats->hist_rgb + _stats_info.histogram_bins);
    return new X3aStatsData (stats);
}

SmartPtr<BufferProxy>
X3aStatsPool::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    SmartPtr<X3aStatsData> stats_data = data.dynamic_cast_ptr<X3aStatsData> ();
    XCAM_ASSERT (stats_data.ptr ());

    return new X3aStats (stats_data);
}

};
