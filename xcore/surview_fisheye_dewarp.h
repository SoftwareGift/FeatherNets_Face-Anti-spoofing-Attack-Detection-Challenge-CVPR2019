/*
 * surview_fisheye_dewarp.h - dewarp fisheye image for surround view
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

#ifndef XCAM_SURVIEW_FISHEYE_DEWARP_H
#define XCAM_SURVIEW_FISHEYE_DEWARP_H

#include <xcam_std.h>
#include <vec_mat.h>
#include <interface/data_types.h>

namespace XCam {

class SurViewFisheyeDewarp
{

public:
    typedef std::vector<PointFloat2> MapTable;

    explicit SurViewFisheyeDewarp ();
    virtual ~SurViewFisheyeDewarp ();

    void fisheye_dewarp(MapTable &map_table, uint32_t table_w, uint32_t table_h, uint32_t image_w, uint32_t image_h, const BowlDataConfig &bowl_config);

    void set_intrinsic_param(const IntrinsicParameter &intrinsic_param);
    void set_extrinsic_param(const ExtrinsicParameter &extrinsic_param);

    IntrinsicParameter get_intrinsic_param();
    ExtrinsicParameter get_extrinsic_param();

private:
    XCAM_DEAD_COPY (SurViewFisheyeDewarp);

    virtual void cal_image_coord (const PointFloat3 &cam_coord, PointFloat2 &image_coord);

    void cal_cam_world_coord (const PointFloat3 &world_coord, PointFloat3 &cam_world_coord);
    void world_coord2cam (const PointFloat3 &cam_world_coord, PointFloat3 &cam_coord);

    Mat4f generate_rotation_matrix(float roll, float pitch, float yaw);

private:
    IntrinsicParameter _intrinsic_param;
    ExtrinsicParameter _extrinsic_param;
};

class PolyFisheyeDewarp : public SurViewFisheyeDewarp
{

public:
    explicit PolyFisheyeDewarp ();

private:
    void cal_image_coord (const PointFloat3 &cam_coord, PointFloat2 &image_coord);

};

} // Adopt Scaramuzza's approach to calculate image coordinates from camera coordinates

#endif // XCAM_SURVIEW_FISHEYE_DEWARP_H
