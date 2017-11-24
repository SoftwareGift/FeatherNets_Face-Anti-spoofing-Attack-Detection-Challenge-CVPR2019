/*
 * xcam_utils.h - xcam utilities
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

#ifndef XCAM_UTILS_H
#define XCAM_UTILS_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <base/xcam_common.h>
#include <xcam_obj_debug.h>
extern "C" {
#include <linux/videodev2.h>
}
#include <cinttypes>
#include <smartptr.h>
#include <vector>

namespace XCam {

static const int64_t InvalidTimestamp = INT64_C(-1);

enum CLWaveletBasis {
    CL_WAVELET_DISABLED = 0,
    CL_WAVELET_HAT,
    CL_WAVELET_HAAR,
};

enum CLImageChannel {
    CL_IMAGE_CHANNEL_Y = 1,
    CL_IMAGE_CHANNEL_UV = 1 << 1,
};

enum CLNV12PlaneIdx {
    CLNV12PlaneY = 0,
    CLNV12PlaneUV,
    CLNV12PlaneMax,
};

double
linear_interpolate_p2 (
    double value_start, double value_end,
    double ref_start, double ref_end,
    double ref_curr);

double
linear_interpolate_p4(
    double value_lt, double value_rt,
    double value_lb, double value_rb,
    double ref_lt_x, double ref_rt_x,
    double ref_lb_x, double ref_rb_x,
    double ref_lt_y, double ref_rt_y,
    double ref_lb_y, double ref_rb_y,
    double ref_curr_x, double ref_curr_y);

void get_gauss_table (
    uint32_t radius, float sigma, std::vector<float> &table, bool normalize = true);

class VideoBuffer;
void dump_buf_perfix_path (const SmartPtr<VideoBuffer> buf, const char *prefix_name);
bool dump_video_buf (const SmartPtr<VideoBuffer> buf, const char *file_name);

};

#endif //XCAM_UTILS_H
