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

inline double
linear_interpolate_p2 (double value_start, double value_end,
                       double ref_start, double ref_end,
                       double ref_curr)
{
    double weight_start = 0;
    double weight_end = 0;
    double dist_start = 0;
    double dist_end = 0;
    double dist_sum = 0;
    double value = 0;

    dist_start = abs(ref_curr - ref_start);
    dist_end = abs(ref_end - ref_curr);
    dist_sum = dist_start + dist_end;

    if (dist_start == 0) {
        weight_start = 10000000.0;
    } else {
        weight_start = ((double)dist_sum / dist_start);
    }

    if (dist_end == 0) {
        weight_end = 10000000.0;
    } else {
        weight_end = ((double)dist_sum / dist_end);
    }

    value = (value_start * weight_start + value_end * weight_end) / (weight_start + weight_end);
    return value;
}

inline double
linear_interpolate_p4(double value_lt, double value_rt,
                      double value_lb, double value_rb,
                      double ref_lt_x, double ref_rt_x,
                      double ref_lb_x, double ref_rb_x,
                      double ref_lt_y, double ref_rt_y,
                      double ref_lb_y, double ref_rb_y,
                      double ref_curr_x, double ref_curr_y)
{
    double weight_lt = 0;
    double weight_rt = 0;
    double weight_lb = 0;
    double weight_rb = 0;
    double dist_lt = 0;
    double dist_rt = 0;
    double dist_lb = 0;
    double dist_rb = 0;
    double dist_sum = 0;
    double value = 0;

    dist_lt = (double)abs(ref_curr_x - ref_lt_x) + (double)abs(ref_curr_y - ref_lt_y);
    dist_rt = (double)abs(ref_curr_x - ref_rt_x) + (double)abs(ref_curr_y - ref_rt_y);
    dist_lb = (double)abs(ref_curr_x - ref_lb_x) + (double)abs(ref_curr_y - ref_lb_y);
    dist_rb = (double)abs(ref_curr_x - ref_rb_x) + (double)abs(ref_curr_y - ref_rb_y);
    dist_sum = dist_lt + dist_rt + dist_lb + dist_rb;

    if (dist_lt == 0) {
        weight_lt = 10000000.0;
    } else {
        weight_lt = ((float)dist_sum / dist_lt);
    }
    if (dist_rt == 0) {
        weight_rt = 10000000.0;
    } else {
        weight_rt = ((float)dist_sum / dist_rt);
    }
    if (dist_lb == 0) {
        weight_lb = 10000000.0;
    } else {
        weight_lb = ((float)dist_sum / dist_lb);
    }
    if (dist_rb == 0) {
        weight_rb = 10000000.0;
    } else {
        weight_rb = ((float)dist_sum / dist_rt);
    }

    value = (double)floor ( (value_lt * weight_lt + value_rt * weight_rt +
                             value_lb * weight_lb + value_rb * weight_rb) /
                            (weight_lt + weight_rt + weight_lb + weight_rb) + 0.5 );
    return value;
}

};

#endif //XCAM_UTILS_H
