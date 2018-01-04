/*
 * cl_video_stabilizer.cpp -  Digital Video Stabilization using IMU (Gyroscope, Accelerometer)
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

#include "cl_video_stabilizer.h"
#include "cl_utils.h"

namespace XCam {

static const XCamKernelInfo kernel_video_stab_warp_info [] = {
    {
        "kernel_image_warp_8_pixel",
#include "kernel_image_warp.clx"
        , 0,
    },
    {
        "kernel_image_warp_1_pixel",
#include "kernel_image_warp.clx"
        , 0,
    }
};

CLVideoStabilizerKernel::CLVideoStabilizerKernel (
    const SmartPtr<CLContext> &context,
    const char *name,
    uint32_t channel,
    SmartPtr<CLImageHandler> &handler)
    : CLImageWarpKernel (context, name, channel, handler)
{
    _handler = handler.dynamic_cast_ptr<CLVideoStabilizer> ();
}

CLVideoStabilizer::CLVideoStabilizer (const SmartPtr<CLContext> &context, const char *name)
    : CLImageWarpHandler (context, name)
{
    _projector = new ImageProjector ();
    _filter_radius = 15;
    _motion_filter = new MotionFilter (_filter_radius, 10);

    CoordinateSystemConv world_to_device (AXIS_X, AXIS_MINUS_Z, AXIS_NONE);
    CoordinateSystemConv device_to_image (AXIS_X, AXIS_Y, AXIS_Y);

    align_coordinate_system (world_to_device, device_to_image);

    _input_frame_id = -1;
    _frame_ts[0] = 0;
    _frame_ts[1] = 0;
    _stabilized_frame_id = -1;
}

SmartPtr<VideoBuffer>
CLVideoStabilizer::get_warp_input_buf ()
{
    XCAM_ASSERT (_input_buf_list.size () >= 1);

    SmartPtr<VideoBuffer> buf = (*_input_buf_list.begin ());
    return buf;
}

bool
CLVideoStabilizer::is_ready ()
{
    return CLImageHandler::is_ready ();
}

void
CLVideoStabilizer::reset_counter ()
{
    XCAM_LOG_DEBUG ("reset video stabilizer counter");

    _input_frame_id = -1;
    _stabilized_frame_id = -1;
    xcam_mem_clear (_frame_ts);
    _device_pose[0].clear ();
    _device_pose[1].clear ();
    _input_buf_list.clear ();
}

XCamReturn
CLVideoStabilizer::execute_done (SmartPtr<VideoBuffer> &output)
{
    if (!_input_buf_list.empty ()) {
        _input_buf_list.pop_front ();
    }

    CLImageWarpHandler::execute_done (output);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLVideoStabilizer::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (input.ptr () && output.ptr ());
    XCAM_UNUSED (output);

    if (_input_buf_list.size () >= 2 * _filter_radius + 1) {
        _input_buf_list.pop_front ();
    }
    _input_buf_list.push_back (input);
    _input_frame_id++;

    const VideoBufferInfo & video_info_in = input->get_video_info ();

    _frame_ts[_input_frame_id % 2] = input->get_timestamp ();

    SmartPtr<DevicePose> data = input->find_typed_metadata<DevicePose> ();
    while (data.ptr ()) {
        _device_pose[_input_frame_id % 2].push_back (data);

        input->remove_metadata (data);

        data = input->find_typed_metadata<DevicePose> ();
    }

    Mat3d homography;
    if (_input_frame_id > 0) {
        homography = analyze_motion (
                         _frame_ts[(_input_frame_id - 1) % 2],
                         _device_pose[(_input_frame_id - 1) % 2],
                         _frame_ts[_input_frame_id % 2],
                         _device_pose[_input_frame_id % 2]);

        if (_motions.size () >= 2 * _filter_radius + 1) {
            _motions.pop_front ();
        }
        _motions.push_back (homography);

        _device_pose[(_input_frame_id - 1) % 2].clear ();
    }

    Mat3d proj_mat;
    XCamDVSResult warp_config;
    if (_input_frame_id >= _filter_radius)
    {
        _stabilized_frame_id = _input_frame_id - _filter_radius;
        int32_t cur_stabilized_pos = XCAM_MIN (_stabilized_frame_id, _filter_radius + 1);

        XCAM_LOG_DEBUG ("input id(%ld), stab id(%ld), cur stab pos(%d), filter r(%d)",
                        _input_frame_id,
                        _stabilized_frame_id,
                        cur_stabilized_pos,
                        _filter_radius);

        proj_mat = stabilize_motion (cur_stabilized_pos, _motions);

        Mat3d proj_inv_mat = proj_mat.inverse ();
        warp_config.frame_id = _stabilized_frame_id;
        warp_config.frame_width = video_info_in.width;
        warp_config.frame_height = video_info_in.height;

        for( int i = 0; i < 3; i++ ) {
            for (int j = 0; j < 3; j++) {
                warp_config.proj_mat[i * 3 + j] = proj_inv_mat(i, j);
            }
        }

        set_warp_config (warp_config);
    } else {
        ret = XCAM_RETURN_BYPASS;
    }

    return ret;
}

XCamReturn
CLVideoStabilizer::set_sensor_calibration (CalibrationParams &params)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (_projector.ptr ()) {
        _projector->set_sensor_calibration (params);
    } else {
        ret = XCAM_RETURN_ERROR_PARAM;
    }

    return ret;
}

XCamReturn
CLVideoStabilizer::set_camera_intrinsics (
    double focal_x,
    double focal_y,
    double offset_x,
    double offset_y,
    double skew)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (_projector.ptr ()) {
        _projector->set_camera_intrinsics(
            focal_x,
            focal_y,
            offset_x,
            offset_y,
            skew);
    } else {
        ret = XCAM_RETURN_ERROR_PARAM;
    }

    return ret;
}

XCamReturn
CLVideoStabilizer::align_coordinate_system (
    CoordinateSystemConv &world_to_device,
    CoordinateSystemConv &device_to_image)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    _world_to_device = world_to_device;
    _device_to_image = device_to_image;

    return ret;
}

XCamReturn
CLVideoStabilizer::set_motion_filter (uint32_t radius, float stdev)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    _filter_radius = radius;

    if (_motion_filter.ptr ()) {
        _motion_filter->set_filters (radius, stdev);
    } else {
        ret = XCAM_RETURN_ERROR_PARAM;
    }

    return ret;
}

Mat3d
CLVideoStabilizer::analyze_motion (
    int64_t frame0_ts,
    DevicePoseList pose0_list,
    int64_t frame1_ts,
    DevicePoseList pose1_list)
{
    if (pose0_list.empty () || pose1_list.empty () || !_projector.ptr ()) {
        return Mat3d ();
    }
    XCAM_ASSERT (frame0_ts < frame1_ts);

    Mat3d ext0 = _projector->calc_camera_extrinsics (frame0_ts, pose0_list);

    Mat3d ext1 = _projector->calc_camera_extrinsics (frame1_ts, pose1_list);

    Mat3d extrinsic0 = _projector->align_coordinate_system (
                           _world_to_device,
                           ext0,
                           _device_to_image);

    Mat3d extrinsic1 = _projector->align_coordinate_system (
                           _world_to_device,
                           ext1,
                           _device_to_image);

    return _projector->calc_projective (extrinsic0, extrinsic1);
}

Mat3d
CLVideoStabilizer::stabilize_motion (int32_t stab_frame_id, std::list<Mat3d> &motions)
{
    if (_motion_filter.ptr ()) {
        return _motion_filter->stabilize (stab_frame_id, motions, _input_frame_id);
    } else {
        return Mat3d ();
    }
}

static SmartPtr<CLVideoStabilizerKernel>
create_kernel_video_stab (
    const SmartPtr<CLContext> &context,
    uint32_t channel,
    SmartPtr<CLImageHandler> handler)
{
    SmartPtr<CLVideoStabilizerKernel> stab_kernel;

    const char *name = (channel == CL_IMAGE_CHANNEL_Y ? "kernel_image_warp_y" : "kernel_image_warp_uv");
    char build_options[1024];
    xcam_mem_clear (build_options);

    snprintf (build_options, sizeof (build_options),
              " -DWARP_Y=%d ",
              (channel == CL_IMAGE_CHANNEL_Y ? 1 : 0));

    stab_kernel = new CLVideoStabilizerKernel (context, name, channel, handler);
    XCAM_ASSERT (stab_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, stab_kernel->build_kernel (kernel_video_stab_warp_info[KernelImageWarp], build_options) == XCAM_RETURN_NO_ERROR,
        NULL, "build video stab kernel failed");
    XCAM_ASSERT (stab_kernel->is_valid ());

    return stab_kernel;
}

SmartPtr<CLImageHandler>
create_cl_video_stab_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLImageHandler> video_stab;
    SmartPtr<CLImageKernel> stab_kernel;

    video_stab = new CLVideoStabilizer (context);
    XCAM_ASSERT (video_stab.ptr ());

    stab_kernel = create_kernel_video_stab (context, CL_IMAGE_CHANNEL_Y, video_stab);
    XCAM_ASSERT (stab_kernel.ptr ());
    video_stab->add_kernel (stab_kernel);

    stab_kernel = create_kernel_video_stab (context, CL_IMAGE_CHANNEL_UV, video_stab);
    XCAM_ASSERT (stab_kernel.ptr ());
    video_stab->add_kernel (stab_kernel);

    return video_stab;
}

MotionFilter::MotionFilter (uint32_t radius, float stdev)
    : _radius (radius),
      _stdev (stdev)
{
    set_filters (radius, stdev);
}

MotionFilter::~MotionFilter ()
{
    _weight.clear ();
}

void
MotionFilter::set_filters (uint32_t radius, float stdev)
{
    _radius = radius;
    _stdev = stdev > 0.f ? stdev : std::sqrt (static_cast<float>(radius));

    int scale = 2 * _radius + 1;
    float dis = 0.0f;
    float sum = 0.0f;

    _weight.resize (2 * _radius + 1);

    for (int i = 0; i < scale; i++) {
        dis = ((float)i - radius) * ((float)i - radius);
        _weight[i] = exp(-dis / (_stdev * _stdev));
        sum += _weight[i];
    }

    for (int i = 0; i <= scale; i++) {
        _weight[i] /= sum;
    }

}

Mat3d
MotionFilter::cumulate_motion (uint32_t index, uint32_t from, std::list<Mat3d> &motions)
{
    Mat3d motion;
    motion.eye ();

    uint32_t id = 0;
    std::list<Mat3d>::iterator it;

    if (from < index) {
        for (id = 0, it = motions.begin (); it != motions.end (); id++, ++it) {
            if (from <= id && id < index) {
                motion = (*it) * motion;
            }
        }
        motion = motion.inverse ();
    } else if (from > index) {
        for (id = 0, it = motions.begin (); it != motions.end (); id++, ++it) {
            if (index <= id && id < from) {
                motion = (*it) * motion;
            }
        }
    }

    return motion;
}

Mat3d
MotionFilter::stabilize (int32_t index,
                         std::list<Mat3d> &motions,
                         int32_t max)
{
    Mat3d res;
    res.zeros ();

    double sum = 0.0f;
    int32_t idx_min = XCAM_MAX ((index - _radius), 0);
    int32_t idx_max = XCAM_MIN ((index + _radius), max);

    for (int32_t i = idx_min; i <= idx_max; ++i)
    {
        res = res + cumulate_motion (index, i, motions) * _weight[i];
        sum += _weight[i];
    }
    if (sum > 0.0f) {
        return res * (1 / sum);
    }
    else {
        return Mat3d ();
    }
}

}
