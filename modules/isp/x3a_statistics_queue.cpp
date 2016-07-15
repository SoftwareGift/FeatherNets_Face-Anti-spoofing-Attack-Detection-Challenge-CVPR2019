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
#include <math.h>

namespace XCam {

X3aIspStatsData::X3aIspStatsData (struct atomisp_3a_statistics *isp_data, XCam3AStats *data)
    : X3aStatsData (data)
    , _isp_data (isp_data)
{
    XCAM_ASSERT (_isp_data);
}

X3aIspStatsData::~X3aIspStatsData ()
{
    if (_isp_data) {
        if (_isp_data->data)
            xcam_free (_isp_data->data);
        if (_isp_data->rgby_data)
            xcam_free (_isp_data->rgby_data);
        xcam_free (_isp_data);
    }
}

bool
X3aIspStatsData::fill_standard_stats ()
{
    XCam3AStats *standard_stats = get_stats ();

    XCAM_ASSERT (_isp_data && _isp_data->data);
    XCAM_ASSERT (standard_stats);
    XCAM_FAIL_RETURN (
        WARNING,
        _isp_data && _isp_data->data && standard_stats,
        false,
        "X3aIspStatsData fill standard stats failed with null data allocated");

    const struct atomisp_grid_info &isp_info = _isp_data->grid_info;
    const XCam3AStatsInfo &standard_info = standard_stats->info;
    const struct atomisp_3a_output *isp_data = _isp_data->data;
    XCamGridStat *standard_data = standard_stats->stats;
    uint32_t pixel_count = isp_info.bqs_per_grid_cell * isp_info.bqs_per_grid_cell;
    uint32_t bit_shift = isp_info.elem_bit_depth - 8;

    XCAM_ASSERT (isp_info.width == standard_info.width);
    XCAM_ASSERT (isp_info.height == standard_info.height);
    for (uint32_t i = 0; i < isp_info.height; ++i) {
        for (uint32_t j = 0; j < isp_info.width; ++j) {
            standard_data[i * standard_info.aligned_width + j].avg_y =
                ((isp_data[i * isp_info.aligned_width + j].ae_y / pixel_count) >> bit_shift);
            standard_data[i * standard_info.aligned_width + j].avg_r =
                ((isp_data[i * isp_info.aligned_width + j].awb_r / pixel_count) >> bit_shift);
            standard_data[i * standard_info.aligned_width + j].avg_gr =
                ((isp_data[i * isp_info.aligned_width + j].awb_gr / pixel_count) >> bit_shift);
            standard_data[i * standard_info.aligned_width + j].avg_gb =
                ((isp_data[i * isp_info.aligned_width + j].awb_gb / pixel_count) >> bit_shift);
            standard_data[i * standard_info.aligned_width + j].avg_b =
                ((isp_data[i * isp_info.aligned_width + j].awb_b / pixel_count) >> bit_shift);
            standard_data[i * standard_info.aligned_width + j].valid_wb_count =
                isp_data[i * isp_info.aligned_width + j].awb_cnt;
            standard_data[i * standard_info.aligned_width + j].f_value1 =
                ((isp_data[i * isp_info.aligned_width + j].af_hpf1 / pixel_count) >> bit_shift);
            standard_data[i * standard_info.aligned_width + j].f_value2 =
                ((isp_data[i * isp_info.aligned_width + j].af_hpf2 / pixel_count) >> bit_shift);
        }
    }

    if (isp_info.has_histogram) {
        uint32_t hist_bins = standard_info.histogram_bins;
        // TODO: atom isp hard code histogram to 256 bins
        XCAM_ASSERT (hist_bins == 256);

        XCamHistogram *hist_rgb = standard_stats->hist_rgb;
        uint32_t *hist_y = standard_stats->hist_y;
        const struct atomisp_3a_rgby_output *isp_hist = _isp_data->rgby_data;
        for (uint32_t i = 0; i < hist_bins; i++) {
            hist_rgb[i].r = isp_hist[i].r;
            hist_rgb[i].gr = isp_hist[i].g;
            hist_rgb[i].gb = isp_hist[i].g;
            hist_rgb[i].b = isp_hist[i].b;
            hist_y[i] = isp_hist[i].y;
        }
    }

    return true;
}

X3aIspStatistics::X3aIspStatistics (const SmartPtr<X3aIspStatsData> &stats_data)
    : X3aStats (SmartPtr<X3aStatsData> (stats_data))
{
}

X3aIspStatistics::~X3aIspStatistics ()
{
}

struct atomisp_3a_statistics *
X3aIspStatistics::get_isp_stats ()
{
    SmartPtr<X3aIspStatsData> stats = get_buffer_data ().dynamic_cast_ptr<X3aIspStatsData> ();

    XCAM_FAIL_RETURN(
        WARNING,
        stats.ptr(),
        NULL,
        "X3aIspStatistics get_stats failed with NULL");

    return stats->get_isp_stats ();
}

bool
X3aIspStatistics::fill_standard_stats ()
{
    SmartPtr<X3aIspStatsData> stats = get_buffer_data ().dynamic_cast_ptr<X3aIspStatsData> ();

    XCAM_FAIL_RETURN(
        WARNING,
        stats.ptr(),
        false,
        "X3aIspStatistics fill standard stats failed with NULL stats data");

    return stats->fill_standard_stats ();
}

X3aStatisticsQueue::X3aStatisticsQueue()
{
    xcam_mem_clear (_grid_info);
}

X3aStatisticsQueue::~X3aStatisticsQueue()
{
}

void
X3aStatisticsQueue::set_grid_info (const struct atomisp_grid_info &info)
{
    XCam3AStatsInfo stats_info;

    xcam_mem_clear (stats_info);
    _grid_info = info;

    stats_info.width = info.width;
    stats_info.height = info.height;
    stats_info.aligned_width = info.aligned_width;
    stats_info.aligned_height = info.aligned_height;
    stats_info.grid_pixel_size = info.bqs_per_grid_cell * 2;
    stats_info.bit_depth = 8;
    stats_info.histogram_bins = 256;

    set_stats_info (stats_info);
}

struct atomisp_3a_statistics *
X3aStatisticsQueue::alloc_isp_statsictics ()
{
    XCAM_ASSERT (_grid_info.width && _grid_info.height);
    XCAM_ASSERT (_grid_info.aligned_width && _grid_info.aligned_height);

    uint32_t grid_size = _grid_info.aligned_width * _grid_info.aligned_height;
    //uint32_t grid_size = _grid_info.width * _grid_info.height;

    struct atomisp_3a_statistics *stats = xcam_malloc0_type (struct atomisp_3a_statistics);
    XCAM_ASSERT (stats);
    stats->data = (struct atomisp_3a_output*)xcam_malloc0 (grid_size * sizeof(*stats->data));
    XCAM_ASSERT (stats->data);
    if (!stats || !stats->data)
        return NULL;

    if (_grid_info.has_histogram) {
        // TODO: atom isp hard code histogram to 256 bins
        stats->rgby_data =
            (struct atomisp_3a_rgby_output*)xcam_malloc0 (256 * sizeof(*stats->rgby_data));
        XCAM_ASSERT (stats->rgby_data);
        if (!stats->rgby_data)
            return NULL;
    }

    stats->grid_info = _grid_info;
    return stats;
}

bool
X3aStatisticsQueue::fixate_video_info (VideoBufferInfo &info)
{
    X3aStatsPool::fixate_video_info (info);

    XCam3AStatsInfo &stats_info = get_stats_info ();

    _grid_info.enable = 1;
    _grid_info.use_dmem = 0;
    _grid_info.has_histogram = 0;
    _grid_info.width = stats_info.width;
    _grid_info.height = stats_info.height;
    _grid_info.aligned_width = stats_info.aligned_width;
    _grid_info.aligned_height = stats_info.aligned_height;
    _grid_info.bqs_per_grid_cell = stats_info.grid_pixel_size / 2;
    _grid_info.deci_factor_log2 = (uint32_t)log2 (_grid_info.bqs_per_grid_cell);
    _grid_info.elem_bit_depth = stats_info.bit_depth;

    return X3aStatsPool::fixate_video_info (info);
}

SmartPtr<BufferData>
X3aStatisticsQueue::allocate_data (const VideoBufferInfo &buffer_info)
{
    XCAM_UNUSED (buffer_info);

    XCam3AStats *stats = NULL;
    XCam3AStatsInfo stats_info = get_stats_info ();
    struct atomisp_3a_statistics *isp_stats = alloc_isp_statsictics ();

    stats = (XCam3AStats *) xcam_malloc0 (
                sizeof (XCam3AStats) +
                sizeof (XCamHistogram) * stats_info.histogram_bins +
                sizeof (uint32_t) * stats_info.histogram_bins +
                sizeof (XCamGridStat) * stats_info.aligned_width * stats_info.aligned_height);
    XCAM_ASSERT (isp_stats && stats);
    stats->info = stats_info;
    stats->hist_rgb = (XCamHistogram *) (stats->stats +
                                         stats_info.aligned_width * stats_info.aligned_height);
    stats->hist_y = (uint32_t *) (stats->hist_rgb + stats_info.histogram_bins);

    return new X3aIspStatsData (isp_stats, stats);
}

SmartPtr<BufferProxy>
X3aStatisticsQueue::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    SmartPtr<X3aIspStatsData> stats_data = data.dynamic_cast_ptr<X3aIspStatsData> ();
    XCAM_ASSERT (stats_data.ptr ());

    return new X3aIspStatistics (stats_data);
}

};
