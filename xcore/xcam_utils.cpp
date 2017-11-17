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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "xcam_utils.h"
#include "video_buffer.h"
#include "image_file_handle.h"

namespace XCam {

double
linear_interpolate_p2 (
    double value_start, double value_end,
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

double
linear_interpolate_p4(
    double value_lt, double value_rt,
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

void
get_gauss_table (uint32_t radius, float sigma, std::vector<float> &table, bool normalize)
{
    uint32_t i;
    uint32_t scale = radius * 2 + 1;
    float dis = 0.0f, sum = 1.0f;

    //XCAM_ASSERT (scale < 512);
    table.resize (scale);
    table[radius] = 1.0f;

    for (i = 0; i < radius; i++)  {
        dis = ((float)i - radius) * ((float)i - radius);
        table[i] = table[scale - i - 1] = exp(-dis / (2.0f * sigma * sigma));
        sum += table[i] * 2.0f;
    }

    if (!normalize)
        return;

    for(i = 0; i < scale; i++)
        table[i] /= sum;
}

void
dump_buf_perfix_path (const SmartPtr<VideoBuffer> buf, const char *prefix_name)
{
    char file_name[256];
    XCAM_ASSERT (prefix_name);
    XCAM_ASSERT (buf.ptr ());

    const VideoBufferInfo &info = buf->get_video_info ();
    snprintf (
        file_name, 256, "%s-%dx%d.%s",
        prefix_name, info.width, info.height, xcam_fourcc_to_string (info.format));
    dump_video_buf (buf, file_name);
}

void
dump_video_buf (const SmartPtr<VideoBuffer> buf, const char *file_name)
{
    XCAM_ASSERT (file_name);
    XCAM_ASSERT (buf.ptr ());
    ImageFileHandle writer (file_name, "wb");
    writer.write_buf (buf);
    writer.close ();
}

}
