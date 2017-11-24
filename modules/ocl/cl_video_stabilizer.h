/*
 * cl_video_stabilizer.h -  Digital Video Stabilization using IMU (Gyroscope, Accelerometer)
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

#ifndef XCAM_CL_VIDEO_STABILIZER_H
#define XCAM_CL_VIDEO_STABILIZER_H

#include <xcam_std.h>
#include <meta_data.h>
#include <vec_mat.h>
#include <image_projector.h>
#include <ocl/cl_image_warp_handler.h>

namespace XCam {

class MotionFilter;
class ImageProjector;
class CLVideoStabilizer;
class CLImageWarpKernel;
class CLImageWarpHandler;

class CLVideoStabilizerKernel
    : public CLImageWarpKernel
{
public:
    explicit CLVideoStabilizerKernel (
        const SmartPtr<CLContext> &context,
        const char *name,
        uint32_t channel,
        SmartPtr<CLImageHandler> &handler);

private:
    XCAM_DEAD_COPY (CLVideoStabilizerKernel);

    SmartPtr<CLVideoStabilizer> _handler;
};

class CLVideoStabilizer
    : public CLImageWarpHandler
{
    typedef std::list<SmartPtr<VideoBuffer>> CLImageBufferList;

public:
    explicit CLVideoStabilizer (
        const SmartPtr<CLContext> &context,
        const char *name = "CLVideoStabilizer");

    virtual ~CLVideoStabilizer () {
        _input_buf_list.clear ();
    }

    virtual SmartPtr<VideoBuffer> get_warp_input_buf ();

    virtual bool is_ready ();

    void reset_counter ();

    XCamReturn set_sensor_calibration (CalibrationParams &params);
    XCamReturn set_camera_intrinsics (
        double focal_x,
        double focal_y,
        double offset_x,
        double offset_y,
        double skew);

    XCamReturn align_coordinate_system (
        CoordinateSystemConv& world_to_device,
        CoordinateSystemConv& device_to_image);

    XCamReturn set_motion_filter (uint32_t radius, float stdev);
    uint32_t filter_radius () const {
        return _filter_radius;
    };

    Mat3d analyze_motion (
        int64_t frame0_ts,
        DevicePoseList pose0_list,
        int64_t frame1_ts,
        DevicePoseList pose1_list);

    Mat3d stabilize_motion (int32_t stab_frame_id, std::list<Mat3d> &motions);

protected:
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLVideoStabilizer);

private:
    Mat3d                    _intrinsics;
    CalibrationParams        _calib_params;
    SmartPtr<ImageProjector> _projector;
    SmartPtr<MotionFilter>   _motion_filter;
    CoordinateSystemConv     _world_to_device;
    CoordinateSystemConv     _device_to_image;
    int64_t                  _input_frame_id;
    int64_t                  _frame_ts[2];
    int64_t                  _stabilized_frame_id;
    DevicePoseList           _device_pose[2];
    std::list<Mat3d>         _motions; //motions[i] calculated from frame i to i+1
    uint32_t                 _filter_radius;
    CLImageBufferList        _input_buf_list;
};

SmartPtr<CLImageHandler>
create_cl_video_stab_handler (const SmartPtr<CLContext> &context);


class MotionFilter
{
public:
    MotionFilter (uint32_t radius = 15, float stdev = 10);
    virtual ~MotionFilter ();

    void set_filters (uint32_t radius, float stdev);

    uint32_t radius () const {
        return _radius;
    };
    float stdev () const {
        return _stdev;
    };

    Mat3d stabilize (int32_t index,
                     std::list<Mat3d> &motions,
                     int32_t max);

protected:
    Mat3d cumulate_motion (uint32_t index, uint32_t from, std::list<Mat3d> &motions);

private:
    XCAM_DEAD_COPY (MotionFilter);

private:
    int32_t            _radius;
    float              _stdev;
    std::vector<float> _weight;
};

}
#endif
