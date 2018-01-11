/*
 * xcam_utils.h - xcam utilities
 *
 *  Copyright (c) 2014-2017 Intel Corporation
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

#include <xcam_std.h>
#include <interface/data_types.h>

namespace XCam {

PointFloat2 bowl_view_coords_to_image (
    const BowlDataConfig &config,
    const PointFloat3 &bowl_pos,
    const uint32_t img_width, const uint32_t img_height);

PointFloat3 bowl_view_image_to_world (
    const BowlDataConfig &config,
    const uint32_t img_width, const uint32_t img_height,
    const PointFloat2 &img_pos);

void centralize_bowl_coord_from_cameras (
    ExtrinsicParameter &front_cam, ExtrinsicParameter &right_cam,
    ExtrinsicParameter &rear_cam, ExtrinsicParameter &left_cam,
    PointFloat3 &bowl_coord_offset);

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

SmartPtr<VideoBuffer>
external_buf_to_once_map_buf (
    uint8_t* buf, uint32_t format,
    uint32_t width, uint32_t height,
    uint32_t aligned_width, uint32_t aligned_height,
    uint32_t size);

};

#endif //XCAM_UTILS_H
