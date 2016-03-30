/*
 * xcam_3a_stats.h - 3a stats standard version
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

#ifndef C_XCAM_3A_STATS_H
#define C_XCAM_3A_STATS_H

#include <base/xcam_common.h>

XCAM_BEGIN_DECLARE

typedef struct _XCam3AStatsInfo {
    uint32_t width;
    uint32_t height;
    uint32_t aligned_width;
    uint32_t aligned_height;
    uint32_t grid_pixel_size;  // in pixel
    uint32_t bit_depth;
    uint32_t histogram_bins;

    uint32_t reserved[2];
} XCam3AStatsInfo;

typedef struct _XCamHistogram {
    uint32_t r;
    uint32_t gr;
    uint32_t gb;
    uint32_t b;
} XCamHistogram;

typedef struct _XCamGridStat {
    /* ae */
    uint32_t avg_y;

    /* awb */
    uint32_t avg_r;
    uint32_t avg_gr;
    uint32_t avg_gb;
    uint32_t avg_b;
    uint32_t valid_wb_count;

    /* af */
    uint32_t f_value1;
    uint32_t f_value2;
} XCamGridStat;

typedef struct _XCam3AStats {
    XCam3AStatsInfo info;
    XCamHistogram *hist_rgb;
    uint32_t *hist_y;
    XCamGridStat stats[0];
} XCam3AStats;

XCAM_END_DECLARE

#endif //C_XCAM_3A_STATS_H
