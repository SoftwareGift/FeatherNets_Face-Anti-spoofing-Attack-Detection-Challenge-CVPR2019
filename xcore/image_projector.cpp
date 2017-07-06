/*
 * image_projector.cpp - Calculate 2D image projective matrix
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

#include "image_projector.h"

namespace XCam {

ImageProjector::ImageProjector (CalibrationParams &params)
    : _calib_params (params)
{
    set_camera_intrinsics(
        params.focal_x,
        params.focal_y,
        params.offset_x,
        params.offset_y,
        params.skew);
}

ImageProjector::ImageProjector (
    double focal_x,
    double focal_y,
    double offset_x,
    double offset_y,
    double skew)
{
    set_camera_intrinsics(
        focal_x,
        focal_y,
        offset_x,
        offset_y,
        skew);
}

Quaternd
ImageProjector::interp_orientation (
    int64_t frame_ts,
    const std::vector<Vec4d> &orientation,
    const std::vector<int64_t> &orient_ts,
    int& index)
{
    if (orientation.empty () || orient_ts.empty ()) {
        return Quaternd ();
    }

    int count = orient_ts.size ();
    if (count == 1) {
        return Quaternd(orientation[0]);
    }

    int i = index;
    XCAM_ASSERT(0 <= i && i < count);

    while (i >= 0 && orient_ts[i] > frame_ts) {
        i--;
    }
    if (i < 0) return Quaternd (orientation[0]);

    while (i + 1 < count && orient_ts[i + 1] < frame_ts) {
        i++;
    }
    if (i >= count) return Quaternd (orientation[count - 1]);

    index = i;

    double weight_start = (orient_ts[i + 1] - frame_ts) / (orient_ts[i + 1] - orient_ts[i]);
    double weight_end = 1.0f - weight_start;
    XCAM_ASSERT (weight_start >= 0 && weight_start <= 1.0);
    XCAM_ASSERT (weight_end >= 0 && weight_end <= 1.0);

    return Quaternd (orientation[i] * weight_start + orientation[i + 1] * weight_end);
    //return Quaternd (quat[i]).slerp(weight_start, Quaternd (quat[i + 1]));
}

// rotate coordinate system keeps the handedness of original coordinate system unchanged
//
// axis_to_x: defines the axis of the new cooridinate system that
//    coincide with the X axis of the original coordinate system.
// axis_to_y: defines the axis of the new cooridinate system that
//    coincide with the Y axis of the original coordinate system.
//
Mat3d
ImageProjector::rotate_coordinate_system (
    CoordinateAxisType axis_to_x,
    CoordinateAxisType axis_to_y)
{
    Mat3d t_mat;
    if (axis_to_x == AXIS_X && axis_to_y == AXIS_MINUS_Z) {
        t_mat = Mat3d (Vec3d (1, 0, 0),
                       Vec3d (0, 0, 1),
                       Vec3d (0, -1, 0));
    } else if (axis_to_x == AXIS_X && axis_to_y == AXIS_MINUS_Y) {
        t_mat = Mat3d (Vec3d (1, 0, 0),
                       Vec3d (0, -1, 0),
                       Vec3d (0, 0, -1));
    } else if (axis_to_x == AXIS_X && axis_to_y == AXIS_Z) {
        t_mat = Mat3d (Vec3d (1, 0, 0),
                       Vec3d (0, 0, -1),
                       Vec3d (0, 1, 0));
    } else if (axis_to_x == AXIS_MINUS_Z && axis_to_y == AXIS_Y) {
        t_mat = Mat3d (Vec3d (0, 0, 1),
                       Vec3d (0, 1, 0),
                       Vec3d (-1, 0, 0));
    } else if (axis_to_x == AXIS_MINUS_X && axis_to_y == AXIS_Y) {
        t_mat = Mat3d (Vec3d (-1, 0, 0),
                       Vec3d (0, 1, 0),
                       Vec3d (0, 0, -1));
    } else if (axis_to_x == AXIS_Z && axis_to_y == AXIS_Y) {
        t_mat = Mat3d (Vec3d (0, 0, -1),
                       Vec3d (0, 1, 0),
                       Vec3d (1, 0, 0));
    } else if (axis_to_x == AXIS_MINUS_Y && axis_to_y == AXIS_X) {
        t_mat = Mat3d (Vec3d (0, 1, 0),
                       Vec3d (-1, 0, 0),
                       Vec3d (0, 0, 1));
    } else if (axis_to_x == AXIS_MINUS_X && axis_to_y == AXIS_MINUS_Y) {
        t_mat = Mat3d (Vec3d (-1, 0, 0),
                       Vec3d (0, -1, 0),
                       Vec3d (0, 0, 1));
    } else if (axis_to_x == AXIS_Y && axis_to_y == AXIS_MINUS_X) {
        t_mat = Mat3d (Vec3d (0, -1, 0),
                       Vec3d (1, 0, 0),
                       Vec3d (0, 0, 1));
    } else  {
        t_mat = Mat3d ();
    }
    return t_mat;
}

// mirror coordinate system will change the handedness of original coordinate system
//
// axis_mirror: defines the axis that coordinate system mirror on
//
Mat3d
ImageProjector::mirror_coordinate_system (CoordinateAxisType axis_mirror)
{
    Mat3d t_mat;

    switch (axis_mirror) {
    case AXIS_X:
    case AXIS_MINUS_X:
        t_mat = Mat3d (Vec3d (-1, 0, 0),
                       Vec3d (0, 1, 0),
                       Vec3d (0, 0, 1));
        break;
    case AXIS_Y:
    case AXIS_MINUS_Y:
        t_mat = Mat3d (Vec3d (1, 0, 0),
                       Vec3d (0, -1, 0),
                       Vec3d (0, 0, 1));
        break;
    case AXIS_Z:
    case AXIS_MINUS_Z:
        t_mat = Mat3d (Vec3d (1, 0, 0),
                       Vec3d (0, 1, 0),
                       Vec3d (0, 0, -1));
        break;
    default:
        t_mat = Mat3d ();
        break;
    }

    return t_mat;
}

// transform coordinate system will change the handedness of original coordinate system
//
// axis_to_x: defines the axis of the new cooridinate system that
//    coincide with the X axis of the original coordinate system.
// axis_to_y: defines the axis of the new cooridinate system that
//    coincide with the Y axis of the original coordinate system.
// axis_mirror: defines the axis that coordinate system mirror on
Mat3d
ImageProjector::transform_coordinate_system (CoordinateSystemConv &transform)
{
    return mirror_coordinate_system (transform.axis_mirror) *
           rotate_coordinate_system (transform.axis_to_x, transform.axis_to_y);
}

Mat3d
ImageProjector::align_coordinate_system (
    CoordinateSystemConv &world_to_device,
    Mat3d &extrinsics,
    CoordinateSystemConv &device_to_image)
{
    return transform_coordinate_system (world_to_device)
           * extrinsics
           * transform_coordinate_system (device_to_image);
}

XCamReturn
ImageProjector::set_sensor_calibration (CalibrationParams &params)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    _calib_params = params;
    set_camera_intrinsics (
        params.focal_x,
        params.focal_y,
        params.offset_x,
        params.offset_y,
        params.skew);

    return ret;
}

XCamReturn
ImageProjector::set_camera_intrinsics (
    double focal_x,
    double focal_y,
    double offset_x,
    double offset_y,
    double skew)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    _intrinsics = Mat3d (Vec3d (focal_x, 0, 0),
                         Vec3d (skew, focal_y, 0),
                         Vec3d (offset_x, offset_y, 1));

    XCAM_LOG_DEBUG("Intrinsic Matrix(3x3) \n");
    XCAM_LOG_DEBUG("intrinsic = [ %lf, %lf, %lf ; %lf, %lf, %lf ; %lf, %lf, %lf ] \n",
                   _intrinsics(1, 1), _intrinsics(1, 2), _intrinsics(1, 3),
                   _intrinsics(2, 1), _intrinsics(2, 2), _intrinsics(2, 3),
                   _intrinsics(3, 1), _intrinsics(3, 2), _intrinsics(3, 3));
    return ret;
}

Mat3d
ImageProjector::calc_camera_extrinsics (
    const int64_t frame_ts,
    const std::vector<int64_t> &pose_ts,
    const std::vector<Vec4d> &orientation,
    const std::vector<Vec3d> &translation)
{
    if (pose_ts.empty () || orientation.empty () || translation.empty ()) {
        return Mat3d ();
    }

    int index = 0;
    const double ts = frame_ts + _calib_params.gyro_delay;
    Quaternd quat = interp_orientation (ts, orientation, pose_ts, index) +
                    Quaternd (_calib_params.gyro_drift);

    Mat3d extrinsics = quat.rotation_matrix ();

    XCAM_LOG_DEBUG("Extrinsic Matrix(3x3) \n");
    XCAM_LOG_DEBUG("extrinsic = [ %lf, %lf, %lf; %lf, %lf, %lf; %lf, %lf, %lf ] \n",
                   extrinsics(1, 1), extrinsics(1, 2), extrinsics(1, 3),
                   extrinsics(2, 1), extrinsics(2, 2), extrinsics(2, 3),
                   extrinsics(3, 1), extrinsics(3, 2), extrinsics(3, 3));

    return extrinsics;
}

Mat3d
ImageProjector::calc_camera_extrinsics (
    const int64_t frame_ts,
    DevicePoseList &pose_list)
{
    if (pose_list.empty ()) {
        return Mat3d ();
    }

    int index = 0;

    std::vector<Vec4d> orientation;
    std::vector<int64_t> orient_ts;
    std::vector<Vec3d> translation;

    for (DevicePoseList::iterator iter = pose_list.begin (); iter != pose_list.end (); ++iter)
    {
        SmartPtr<DevicePose> pose = *iter;

        orientation.push_back (Vec4d (pose->orientation[0],
                                      pose->orientation[1],
                                      pose->orientation[2],
                                      pose->orientation[3]));

        orient_ts.push_back (pose->timestamp);

        translation.push_back (Vec3d (pose->translation[0],
                                      pose->translation[1],
                                      pose->translation[2]));

    }

    const int64_t ts = frame_ts + _calib_params.gyro_delay;
    Quaternd quat = interp_orientation (ts, orientation, orient_ts, index) +
                    Quaternd (_calib_params.gyro_drift);

    Mat3d extrinsics = quat.rotation_matrix ();

    XCAM_LOG_DEBUG("Extrinsic Matrix(3x3) \n");
    XCAM_LOG_DEBUG("extrinsic = [ %lf, %lf, %lf; %lf, %lf, %lf; %lf, %lf, %lf ] \n",
                   extrinsics(1, 1), extrinsics(1, 2), extrinsics(1, 3),
                   extrinsics(2, 1), extrinsics(2, 2), extrinsics(2, 3),
                   extrinsics(3, 1), extrinsics(3, 2), extrinsics(3, 3));

    return extrinsics;
}

Mat3d
ImageProjector::calc_projective (
    Mat3d &extrinsic0,
    Mat3d &extrinsic1)
{
    Mat3d intrinsic = get_camera_intrinsics ();

    return intrinsic * extrinsic0 * extrinsic1.transpose () * intrinsic.inverse ();
}

}

