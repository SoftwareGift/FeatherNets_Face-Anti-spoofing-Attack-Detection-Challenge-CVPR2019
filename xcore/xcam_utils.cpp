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
 * Author: Junkai Wu <junkai.wu@intel.com>
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "xcam_utils.h"
#include "video_buffer.h"
#include "image_file_handle.h"

namespace XCam {

static float
transform_bowl_coord_to_image_x (
    const float bowl_x, const float bowl_y,
    const uint32_t img_width)
{
    float offset_radian = (bowl_x < 0.0f) ? PI : ((bowl_y >= 0.0f) ? 2.0f * PI : 0.0f);
    float arctan_radian = (bowl_x != 0.0f) ? atan (-bowl_y / bowl_x) : ((bowl_y >= 0.0f) ? -PI / 2.0f : PI / 2.0f);

    float img_x = arctan_radian + offset_radian;
    img_x *= img_width / (2.0f * PI);
    return XCAM_CLAMP (img_x, 0.0f, img_width - 1.0f);
}

static float
transform_bowl_coord_to_image_y (
    const BowlDataConfig &config,
    const float bowl_x, const float bowl_y, const float bowl_z,
    const uint32_t img_height)
{
    float wall_image_height = config.wall_height / (config.wall_height + config.ground_length) * img_height;
    float ground_image_height = img_height - wall_image_height;
    float img_y = 0.0f;

    if (bowl_z > 0.0f) {
        img_y = (config.wall_height - bowl_z) * wall_image_height / config.wall_height;
        img_y = XCAM_CLAMP (img_y, 0.0f, wall_image_height - 1.0f);
    } else {
        float max_semimajor = config.b *
                              sqrt (1 - config.center_z * config.center_z / (config.c * config.c));
        float min_semimajor = max_semimajor - config.ground_length;
        XCAM_ASSERT (min_semimajor >= 0);
        XCAM_ASSERT (max_semimajor > min_semimajor);
        float step = ground_image_height / (max_semimajor - min_semimajor);

        float axis_ratio = config.a / config.b;
        float cur_semimajor = sqrt (bowl_x * bowl_x + bowl_y * bowl_y * axis_ratio * axis_ratio) / axis_ratio;
        cur_semimajor = XCAM_CLAMP (cur_semimajor, min_semimajor, max_semimajor);

        img_y = (max_semimajor - cur_semimajor) * step + wall_image_height;
        img_y = XCAM_CLAMP (img_y, wall_image_height, img_height - 1.0f);
    }
    return img_y;
}

PointFloat2 bowl_view_coords_to_image (
    const BowlDataConfig &config,
    const PointFloat3 &bowl_pos,
    const uint32_t img_width, const uint32_t img_height)
{
    PointFloat2 img_pos;
    img_pos.x = transform_bowl_coord_to_image_x (bowl_pos.x, bowl_pos.y, img_width);
    img_pos.y = transform_bowl_coord_to_image_y (config, bowl_pos.x, bowl_pos.y, bowl_pos.z, img_height);

    return img_pos;
}

PointFloat3 bowl_view_image_to_world (
    const BowlDataConfig &config,
    const uint32_t img_width, const uint32_t img_height,
    const PointFloat2 &img_pos)
{
    PointFloat3 world;
    float angle;

    float a = config.a;
    float b = config.b;
    float c = config.c;

    float wall_image_height = config.wall_height / (float)(config.wall_height + config.ground_length) * (float)img_height;
    float ground_image_height = (float)img_height - wall_image_height;

    float z_step = (float)config.wall_height / wall_image_height;
    float angle_step = fabs(config.angle_end - config.angle_start) / img_width;

    if(img_pos.y < wall_image_height) {
        world.z = config.wall_height - img_pos.y * z_step; // TODO world.z
        angle = degree2radian (config.angle_start + img_pos.x * angle_step);
        float r2 = 1 - (world.z - config.center_z) * (world.z - config.center_z) / (c * c);

        if(XCAM_DOUBLE_EQUAL_AROUND (angle, PI / 2)) {
            world.x = 0.0f;
            world.y = -sqrt(r2 * b * b);
        } else if (XCAM_DOUBLE_EQUAL_AROUND (angle, PI * 3 / 2)) {
            world.x = 0.0f;
            world.y = sqrt(r2 * b * b);
        } else if((angle < PI / 2) || (angle > PI * 3 / 2)) {
            world.x = sqrt(r2 * a * a * b * b / (b * b + a * a * tan(angle) * tan(angle)));
            world.y = -world.x * tan(angle);
        } else {
            world.x = -sqrt(r2 * a * a * b * b / (b * b + a * a * tan(angle) * tan(angle)));
            world.y = -world.x * tan(angle);
        }
    } else {
        a = a * sqrt(1 - config.center_z * config.center_z / (c * c));
        b = b * sqrt(1 - config.center_z * config.center_z / (c * c));

        float ratio_ab = b / a;

        float step_b = config.ground_length / ground_image_height;

        b = b - (img_pos.y - wall_image_height) * step_b;
        a = b / ratio_ab;

        angle = degree2radian (config.angle_start + img_pos.x * angle_step);

        if(XCAM_DOUBLE_EQUAL_AROUND (angle, PI / 2)) {
            world.x = 0.0f;
            world.y = -b;
        } else if (XCAM_DOUBLE_EQUAL_AROUND (angle, PI * 3 / 2)) {
            world.x = 0.0f;
            world.y = b;
        } else if((angle < PI / 2) || (angle > PI * 3 / 2)) {
            world.x = a * b / sqrt(b * b + a * a * tan(angle) * tan(angle));
            world.y = -world.x * tan(angle);
        } else {
            world.x = -a * b / sqrt(b * b + a * a * tan(angle) * tan(angle));
            world.y = -world.x * tan(angle);
        }
        world.z = 0.0f;
    }

    return world;
}

void centralize_bowl_coord_from_cameras (
    ExtrinsicParameter &front_cam, ExtrinsicParameter &right_cam,
    ExtrinsicParameter &rear_cam, ExtrinsicParameter &left_cam,
    PointFloat3 &bowl_coord_offset)
{
    bowl_coord_offset.x = (front_cam.trans_x + rear_cam.trans_x) / 2.0f;
    bowl_coord_offset.y = (right_cam.trans_y + left_cam.trans_y) / 2.0f;
    bowl_coord_offset.z = 0.0f;

    front_cam.trans_x -= bowl_coord_offset.x;
    front_cam.trans_y -= bowl_coord_offset.y;

    right_cam.trans_x -= bowl_coord_offset.x;
    right_cam.trans_y -= bowl_coord_offset.y;

    rear_cam.trans_x -= bowl_coord_offset.x;
    rear_cam.trans_y -= bowl_coord_offset.y;

    left_cam.trans_x -= bowl_coord_offset.x;
    left_cam.trans_y -= bowl_coord_offset.y;
}

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

bool
dump_video_buf (const SmartPtr<VideoBuffer> buf, const char *file_name)
{
    ImageFileHandle file;
    XCAM_ASSERT (file_name);

    XCamReturn ret = file.open (file_name, "wb");
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), false,
        "dump buffer failed when open file: %s", file_name);

    ret = file.write_buf (buf);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), false,
        "dump buffer to file: %s failed", file_name);

    return true;
}

}
