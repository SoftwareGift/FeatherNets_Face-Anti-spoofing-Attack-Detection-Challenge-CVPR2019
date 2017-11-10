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

namespace XCam {

SurViewFisheyeDewarp::SurViewFisheyeDewarp ()
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
    MapTable world_coord(3);
    MapTable cam_coord(3);
    MapTable cam_world_coord(3);
    MapTable image_coord(2);

    uint32_t scale_factor_w = image_w / table_w;
    uint32_t scale_factor_h = image_h / table_h;

    for(uint32_t row = 0; row < table_h; row++) {
        for(uint32_t col = 0; col < table_w; col++) {
            uint32_t x = col * scale_factor_w;
            uint32_t y = row * scale_factor_h;

            cal_world_coord(x, y, world_coord, image_w, image_h, bowl_config);
            cal_cam_world_coord(world_coord, cam_world_coord);
            world_coord2cam(cam_world_coord, cam_coord);
            cal_image_coord(cam_coord, image_coord);

            map_table[row * table_w * 2 + col * 2] = image_coord[0];
            map_table[row * table_w * 2 + col * 2 + 1] = image_coord[1];
        }
    }
}

void
SurViewFisheyeDewarp::cal_world_coord(uint32_t x, uint32_t y, MapTable &world_coord, uint32_t image_w, uint32_t image_h, const BowlDataConfig &bowl_config)
{
    float world_x, world_y, world_z;
    float angle;

    float a = bowl_config.a;
    float b = bowl_config.b;
    float c = bowl_config.c;

    uint32_t wall_image_height = bowl_config.wall_height / (bowl_config.wall_height + bowl_config.ground_length) * image_h;
    uint32_t ground_image_height = image_h - wall_image_height;

    float z_step = bowl_config.wall_height / wall_image_height;
    float angle_step = fabs(bowl_config.angle_end - bowl_config.angle_start) / image_w;

    if(y < wall_image_height) {
        world_z = bowl_config.wall_height - bowl_config.center_z - y * z_step;
        angle = degree2radian (bowl_config.angle_end - x * angle_step);
        float r2 = 1 - world_z * world_z / (c * c);

        if(XCAM_DOUBLE_EQUAL_AROUND (angle, PI / 2)) {
            world_x = 0.0f;
            world_y = sqrt(r2 * b * b);
        } else if (XCAM_DOUBLE_EQUAL_AROUND (angle, PI * 3 / 2)) {
            world_x = 0.0f;
            world_y = -sqrt(r2 * b * b);
        } else if((angle < PI / 2) || (angle > PI * 3 / 2)) {
            world_x = sqrt(r2 * a * a * b * b / (b * b + a * a * tan(angle) * tan(angle)));
            world_y = world_x * tan(angle);
        } else {
            world_x = -sqrt(r2 * a * a * b * b / (b * b + a * a * tan(angle) * tan(angle)));
            world_y = world_x * tan(angle);
        }
    } else {
        world_z = -bowl_config.center_z;
        a = a * sqrt(1 - world_z * world_z / (c * c));
        b = b * sqrt(1 - world_z * world_z / (c * c));

        float ratio_ab = b / a;

        float step_b = bowl_config.ground_length / ground_image_height;

        b = b - (y - wall_image_height) * step_b;
        a = b / ratio_ab;

        angle = degree2radian (bowl_config.angle_end - x * angle_step);

        if(XCAM_DOUBLE_EQUAL_AROUND (angle, PI / 2)) {
            world_x = 0.0f;
            world_y = b;
        } else if (XCAM_DOUBLE_EQUAL_AROUND (angle, PI * 3 / 2)) {
            world_x = 0.0f;
            world_y = -b;
        } else if((angle < PI / 2) || (angle > PI * 3 / 2)) {
            world_x = a * b / sqrt(b * b + a * a * tan(angle) * tan(angle));
            world_y = world_x * tan(angle);
        } else {
            world_x = -a * b / sqrt(b * b + a * a * tan(angle) * tan(angle));
            world_y = world_x * tan(angle);
        }
    }

    world_coord[0] = world_x;
    world_coord[1] = world_y;
    world_coord[2] = world_z + bowl_config.center_z;
}

void
SurViewFisheyeDewarp::cal_cam_world_coord(MapTable world_coord, MapTable &cam_world_coord)
{
    Mat4f rotation_mat = generate_rotation_matrix( degree2radian (_extrinsic_param.roll),
                         degree2radian (_extrinsic_param.pitch),
                         degree2radian (_extrinsic_param.yaw));
    Mat4f rotation_tran_mat = rotation_mat;
    rotation_tran_mat(0, 3) = _extrinsic_param.trans_x;
    rotation_tran_mat(1, 3) = _extrinsic_param.trans_y;
    rotation_tran_mat(2, 3) = _extrinsic_param.trans_z;

    Mat4f world_coord_mat(Vec4f(1.0f, 0.0f, 0.0f, world_coord[0]),
                          Vec4f(0.0f, 1.0f, 0.0f, world_coord[1]),
                          Vec4f(0.0f, 0.0f, 1.0f, world_coord[2]),
                          Vec4f(0.0f, 0.0f, 0.0f, 1.0f));

    Mat4f cam_world_coord_mat = rotation_tran_mat.inverse() * world_coord_mat;

    cam_world_coord[0] = cam_world_coord_mat(0, 3);
    cam_world_coord[1] = cam_world_coord_mat(1, 3);
    cam_world_coord[2] = cam_world_coord_mat(2, 3);
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
SurViewFisheyeDewarp::world_coord2cam(MapTable cam_world_coord, MapTable &cam_coord)
{
    cam_coord[0] = -cam_world_coord[1];
    cam_coord[1] = -cam_world_coord[2];
    cam_coord[2] = -cam_world_coord[0];
}

void
SurViewFisheyeDewarp::cal_image_coord(MapTable cam_coord, MapTable &image_coord)
{
    image_coord[0] = cam_coord[0];
    image_coord[1] = cam_coord[1];
}

void
PolyFisheyeDewarp::cal_image_coord(MapTable cam_coord, MapTable &image_coord)
{
    float dist2center = sqrt(cam_coord[0] * cam_coord[0] + cam_coord[1] * cam_coord[1]);
    float angle = atan(cam_coord[2] / dist2center);

    float p = 1;
    float poly_sum = 0;

    IntrinsicParameter intrinsic_param = get_intrinsic_param();

    if (dist2center != 0) {
        for (uint32_t i = 0; i < intrinsic_param.poly_length; i++) {
            poly_sum += intrinsic_param.poly_coeff[i] * p;
            p = p * angle;
        }

        float image_x = cam_coord[0] * poly_sum / dist2center;
        float image_y = cam_coord[1] * poly_sum / dist2center;

        image_coord[0] = image_x * intrinsic_param.c + image_y * intrinsic_param.d + intrinsic_param.xc;
        image_coord[1] = image_x * intrinsic_param.e + image_y + intrinsic_param.yc;
    } else {
        image_coord[0] = intrinsic_param.xc;
        image_coord[1] = intrinsic_param.yc;
    }
} // Adopt Scaramuzza's approach to calculate image coordinates from camera coordinates

}
