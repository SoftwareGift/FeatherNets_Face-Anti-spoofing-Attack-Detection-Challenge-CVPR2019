/*
 * image_projector.h - Calculate 2D image projective matrix
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_IMAGE_PROJECTIVE_2D_H
#define XCAM_IMAGE_PROJECTIVE_2D_H

#include <xcam_std.h>
#include <meta_data.h>
#include <vec_mat.h>
#include <vector>

namespace XCam {

struct CalibrationParams {
    double focal_x;  //Focal length, x axis, in pixels
    double focal_y;  //Focal length, y axis, in pixels
    double offset_x;  //Principal point x coordinate on the image, in pixels
    double offset_y;  //Principal point y coordinate on the image, in pixels
    double skew; //in case if the image coordinate axes u and v are not orthogonal to each other
    double readout_time;
    double gyro_delay;
    Vec4d gyro_drift;

    CalibrationParams ()
        : focal_x (0)
        , focal_y (0)
        , offset_x (0)
        , offset_y (0)
        , skew (0)
        , readout_time (0)
        , gyro_delay (0)
    {
        gyro_drift.zeros();
    }
};

enum CoordinateAxisType {
    AXIS_X = 0,
    AXIS_MINUS_X,
    AXIS_Y,
    AXIS_MINUS_Y,
    AXIS_Z,
    AXIS_MINUS_Z,
    AXIS_NONE,
};

struct CoordinateSystemConv {
    CoordinateAxisType axis_to_x;
    CoordinateAxisType axis_to_y;
    CoordinateAxisType axis_mirror;

    CoordinateSystemConv ()
    {
        axis_to_x = AXIS_X;
        axis_to_y = AXIS_Y;
        axis_mirror = AXIS_NONE;
    }

    CoordinateSystemConv (
        CoordinateAxisType to_x,
        CoordinateAxisType to_y,
        CoordinateAxisType mirror)
    {
        axis_to_x = to_x;
        axis_to_y = to_y;
        axis_mirror = mirror;
    }
};

class ImageProjector
{
public:
    explicit ImageProjector () {};
    explicit ImageProjector (CalibrationParams &params);
    explicit ImageProjector (
        double focal_x,
        double focal_y,
        double offset_x,
        double offset_y,
        double skew);

    virtual ~ImageProjector () {};

    XCamReturn set_sensor_calibration (CalibrationParams &params);
    XCamReturn set_camera_intrinsics (
        double focal_x,
        double focal_y,
        double offset_x,
        double offset_y,
        double skew);

    Mat3d get_camera_intrinsics () {
        return _intrinsics;
    }

    Mat3d calc_camera_extrinsics (
        const int64_t frame_ts,
        const std::vector<int64_t> &pose_ts,
        const std::vector<Vec4d> &orientation,
        const std::vector<Vec3d> &translation);

    Mat3d calc_camera_extrinsics (
        const int64_t frame_ts,
        DevicePoseList &pose_list);

    Mat3d calc_projective (
        Mat3d &extrinsic0,
        Mat3d &extrinsic1);

    Mat3d align_coordinate_system (
        CoordinateSystemConv &world_to_device,
        Mat3d &extrinsics,
        CoordinateSystemConv &device_to_image);

protected:
    Quaternd interp_orientation (
        int64_t ts,
        const std::vector<Vec4d> &orientation,
        const std::vector<int64_t> &orient_ts,
        int& index);

    Mat3d rotate_coordinate_system (
        CoordinateAxisType axis_to_x,
        CoordinateAxisType axis_to_y);

    Mat3d mirror_coordinate_system (CoordinateAxisType axis_mirror);

    Mat3d transform_coordinate_system (CoordinateSystemConv &transform);

private:
    XCAM_DEAD_COPY (ImageProjector);

private:
    Mat3d             _intrinsics;
    CalibrationParams _calib_params;
};

}

#endif //XCAM_IMAGE_PROJECTIVE_2D_H