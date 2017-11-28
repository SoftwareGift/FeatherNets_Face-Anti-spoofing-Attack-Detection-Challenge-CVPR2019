/*
 * surview_fisheye_dewarp.cpp - dewarp fisheye image of surround view
 *
 *  Copyright (c) 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Junkai Wu <junkai.wu@intel.com>
 */

#include "surview_fisheye_dewarp.h"
#include "xcam_utils.h"

namespace XCam {

SurViewFisheyeDewarp::SurViewFisheyeDewarp ()
{
}
SurViewFisheyeDewarp::~SurViewFisheyeDewarp ()
{
}

PolyFisheyeDewarp::PolyFisheyeDewarp()
    : SurViewFisheyeDewarp()
{
}

void
SurViewFisheyeDewarp::set_intrinsic_param(const IntrinsicParameter &intrinsic_param)
{
    _intrinsic_param = intrinsic_param;
}

void
SurViewFisheyeDewarp::set_extrinsic_param(const ExtrinsicParameter &extrinsic_param)
{
    _extrinsic_param = extrinsic_param;
}

IntrinsicParameter
SurViewFisheyeDewarp::get_intrinsic_param()
{
    return _intrinsic_param;
}

ExtrinsicParameter
SurViewFisheyeDewarp::get_extrinsic_param()
{
    return _extrinsic_param;
}

void
SurViewFisheyeDewarp::fisheye_dewarp(MapTable &map_table, uint32_t table_w, uint32_t table_h, uint32_t image_w, uint32_t image_h, const BowlDataConfig &bowl_config)
{
    PointFloat3 world_coord;
    PointFloat3 cam_coord;
    PointFloat3 cam_world_coord;
    PointFloat2 image_coord;

    XCAM_LOG_DEBUG ("fisheye-dewarp:\n table(%dx%d), out_size(%dx%d)"
                    "bowl(start:%.1f, end:%.1f, ground:%.2f, wall:%.2f, a:%.2f, b:%.2f, c:%.2f, center_z:%.2f )",
                    table_w, table_h, image_w, image_h,
                    bowl_config.angle_start, bowl_config.angle_end,
                    bowl_config.wall_height, bowl_config.ground_length,
                    bowl_config.a, bowl_config.b, bowl_config.c, bowl_config.center_z);

    float scale_factor_w = (float)image_w / table_w;
    float scale_factor_h = (float)image_h / table_h;

    for(uint32_t row = 0; row < table_h; row++) {
        for(uint32_t col = 0; col < table_w; col++) {
            PointFloat2 out_pos (col * scale_factor_w, row * scale_factor_h);
            world_coord = bowl_view_image_to_world (bowl_config, image_w, image_h, out_pos);
            cal_cam_world_coord(world_coord, cam_world_coord);
            world_coord2cam(cam_world_coord, cam_coord);
            cal_image_coord(cam_coord, image_coord);

            map_table[row * table_w + col] = image_coord;
        }
    }
}

void
SurViewFisheyeDewarp::cal_cam_world_coord(const PointFloat3 &world_coord, PointFloat3 &cam_world_coord)
{
    Mat4f rotation_mat = generate_rotation_matrix( degree2radian (_extrinsic_param.roll),
                         degree2radian (_extrinsic_param.pitch),
                         degree2radian (_extrinsic_param.yaw));
    Mat4f rotation_tran_mat = rotation_mat;
    rotation_tran_mat(0, 3) = _extrinsic_param.trans_x;
    rotation_tran_mat(1, 3) = _extrinsic_param.trans_y;
    rotation_tran_mat(2, 3) = _extrinsic_param.trans_z;

    Mat4f world_coord_mat(Vec4f(1.0f, 0.0f, 0.0f, world_coord.x),
                          Vec4f(0.0f, 1.0f, 0.0f, world_coord.y),
                          Vec4f(0.0f, 0.0f, 1.0f, world_coord.z),
                          Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    Mat4f cam_world_coord_mat = rotation_tran_mat.inverse() * world_coord_mat;

    cam_world_coord.x = cam_world_coord_mat(0, 3);
    cam_world_coord.y = cam_world_coord_mat(1, 3);
    cam_world_coord.z = cam_world_coord_mat(2, 3);
}

Mat4f
SurViewFisheyeDewarp::generate_rotation_matrix(float roll, float pitch, float yaw)
{
    Mat4f matrix_x(Vec4f(1.0f, 0.0f, 0.0f, 0.0f),
                   Vec4f(0.0f, cos(roll), -sin(roll), 0.0f),
                   Vec4f(0.0f, sin(roll), cos(roll), 0.0f),
                   Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    Mat4f matrix_y(Vec4f(cos(pitch), 0.0f, sin(pitch), 0.0f),
                   Vec4f(0.0f, 1.0f, 0.0f, 0.0f),
                   Vec4f(-sin(pitch), 0.0f, cos(pitch), 0.0f),
                   Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    Mat4f matrix_z(Vec4f(cos(yaw), -sin(yaw), 0.0f, 0.0f),
                   Vec4f(sin(yaw), cos(yaw), 0.0f, 0.0f),
                   Vec4f(0.0f, 0.0f, 1.0f, 0.0f),
                   Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    return matrix_z * matrix_y * matrix_x;
}

void
SurViewFisheyeDewarp::world_coord2cam(const PointFloat3 &cam_world_coord, PointFloat3 &cam_coord)
{
    cam_coord.x = -cam_world_coord.y;
    cam_coord.y = -cam_world_coord.z;
    cam_coord.z = -cam_world_coord.x;
}

void
SurViewFisheyeDewarp::cal_image_coord(const PointFloat3 &cam_coord, PointFloat2 &image_coord)
{
    image_coord.x = cam_coord.x;
    image_coord.y = cam_coord.y;
}

void
PolyFisheyeDewarp::cal_image_coord(const PointFloat3 &cam_coord, PointFloat2 &image_coord)
{
    float dist2center = sqrt(cam_coord.x * cam_coord.x + cam_coord.y * cam_coord.y);
    float angle = atan(cam_coord.z / dist2center);

    float p = 1;
    float poly_sum = 0;

    IntrinsicParameter intrinsic_param = get_intrinsic_param();

    if (dist2center != 0) {
        for (uint32_t i = 0; i < intrinsic_param.poly_length; i++) {
            poly_sum += intrinsic_param.poly_coeff[i] * p;
            p = p * angle;
        }

        float image_x = cam_coord.x * poly_sum / dist2center;
        float image_y = cam_coord.y * poly_sum / dist2center;

        image_coord.x = image_x * intrinsic_param.c + image_y * intrinsic_param.d + intrinsic_param.xc;
        image_coord.y = image_x * intrinsic_param.e + image_y + intrinsic_param.yc;
    } else {
        image_coord.x = intrinsic_param.xc;
        image_coord.y = intrinsic_param.yc;
    }
} // Adopt Scaramuzza's approach to calculate image coordinates from camera coordinates

}
