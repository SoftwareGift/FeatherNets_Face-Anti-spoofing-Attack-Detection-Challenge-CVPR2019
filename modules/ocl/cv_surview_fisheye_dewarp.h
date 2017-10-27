/*
 * cv_surview_fisheye_dewarp.h - dewarp fisheye image for surround view
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

#ifndef XCAM_CV_SURVIEW_FISHEYE_DEWARP_H
#define XCAM_CV_SURVIEW_FISHEYE_DEWARP_H

#include <base/xcam_common.h>
#include <base/xcam_buffer.h>
#include <dma_video_buffer.h>
#include <smartptr.h>
#include "xcam_obj_debug.h"
#include "image_file_handle.h"
#include "cv_base_class.h"
#include "cv_calibration_parser.h"

#include <ocl/cl_context.h>
#include <ocl/cl_device.h>
#include <ocl/cl_memory.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace XCam {

class BowlDataConfig
{

public:
    explicit BowlDataConfig();

    uint32_t wall_image_height;
    uint32_t ground_image_height;

    float a, b, c;
    float angle_start, angle_end;

};

class CVSurViewFisheyeDewarp : public CVBaseClass
{

public:
    typedef std::vector<float> MapTable;

    explicit CVSurViewFisheyeDewarp ();

    void fisheye_dewarp(MapTable &map_table, uint32_t table_w, uint32_t table_h, uint32_t image_w, uint32_t image_h, const BowlDataConfig &bowl_config);

    void set_intrinsic_param(const IntrinsicParameter &intrinsic_param);
    void set_extrinsic_param(const ExtrinsicParameter &extrinsic_param);

    IntrinsicParameter get_intrinsic_param();
    ExtrinsicParameter get_extrinsic_param();

private:
    XCAM_DEAD_COPY (CVSurViewFisheyeDewarp);

    virtual void cal_image_coord(MapTable cam_coord, MapTable &image_coord);

    void cal_world_coord(uint32_t x, uint32_t y, MapTable &world_coord, uint32_t image_w, const BowlDataConfig &bowl_config);
    void cal_cam_world_coord(MapTable world_coord, MapTable &cam_world_coord);
    void world_coord2cam(MapTable cam_world_coord, MapTable &cam_coord);

    cv::Matx44f generate_rotation_matrix(float roll, float pitch, float yaw);

private:
    IntrinsicParameter _intrinsic_param;
    ExtrinsicParameter _extrinsic_param;
};

class CVPolyFisheyeDewarp : public CVSurViewFisheyeDewarp
{

public:
    explicit CVPolyFisheyeDewarp ();

    void cal_image_coord(MapTable cam_coord, MapTable &image_coord);

};

} // Adopt Scaramuzza's approach to calculate image coordinates from camera coordinates

#endif // XCAM_CV_SURVIEW_FISHEYE_DEWARP_H
