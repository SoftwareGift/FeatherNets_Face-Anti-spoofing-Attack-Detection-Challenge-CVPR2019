/*
 * cl_fisheye_handler.h - CL fisheye handler
 *
 *  Copyright (c) 2016 Intel Corporation
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

#ifndef XCAM_CL_FISHEYE_HANDLER_H
#define XCAM_CL_FISHEYE_HANDLER_H

#include <xcam_std.h>
#include <interface/data_types.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_geo_map_handler.h>
#include <surview_fisheye_dewarp.h>

namespace XCam {

class CLFisheyeHandler;
class CLFisheye2GPSKernel
    : public CLImageKernel
{
public:
    explicit CLFisheye2GPSKernel (const SmartPtr<CLContext> &context, SmartPtr<CLFisheyeHandler> &handler);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    SmartPtr<CLFisheyeHandler>  _handler;
};

class CLFisheyeHandler
    : public CLImageHandler
    , public GeoKernelParamCallback
{
    friend class CLFisheye2GPSKernel;
public:
    explicit CLFisheyeHandler (const SmartPtr<CLContext> &context, SurroundMode surround_mode, bool use_map, bool need_lsc, bool need_scale);
    virtual ~CLFisheyeHandler();

    void set_output_size (uint32_t width, uint32_t height);
    void get_output_size (uint32_t &width, uint32_t &height) const;

    void set_dst_range (float longitude, float latitude);
    void get_dst_range (float &longitude, float &latitude) const;
    void set_fisheye_info (const FisheyeInfo &info);
    const FisheyeInfo &get_fisheye_info () const {
        return _fisheye_info;
    }

    void set_lsc_table (float *table, uint32_t table_size);
    void set_lsc_gray_threshold (float min_threshold, float max_threshold);

    void set_stable_y_start (float y_start);
    virtual float get_stable_y_start () {
        return _stable_y_start;
    }

    void set_left_scale_factor (PointFloat2 factor);
    void set_right_scale_factor (PointFloat2 factor);

    virtual PointFloat2 get_left_scale_factor () {
        return _left_scale_factor;
    }
    virtual PointFloat2 get_right_scale_factor () {
        return _right_scale_factor;
    }

    void set_bowl_config(const BowlDataConfig bowl_data_config) {
        _bowl_data_config = bowl_data_config;
    }

    const BowlDataConfig &get_bowl_config() {
        return _bowl_data_config;
    }

    void set_intrinsic_param(const IntrinsicParameter intrinsic_param) {
        _intrinsic_param = intrinsic_param;
    }
    const IntrinsicParameter &get_intrinsic_param() {
        return _intrinsic_param;
    }

    void set_extrinsic_param(const ExtrinsicParameter extrinsic_param) {
        _extrinsic_param = extrinsic_param;
    }
    const ExtrinsicParameter &get_extrinsic_param() {
        return _extrinsic_param;
    }


protected:
    // derived from CLImageHandler
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input, VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

    // derived from GeoKernelParamCallback
    virtual SmartPtr<CLImage> get_geo_input_image (NV12PlaneIdx index);
    virtual SmartPtr<CLImage> get_geo_output_image (NV12PlaneIdx index);
    virtual SmartPtr<CLImage> get_geo_map_table () {
        return _geo_table;
    }
    virtual void get_geo_equivalent_out_size (float &width, float &height);
    virtual void get_geo_pixel_out_size (float &width, float &height);

    virtual SmartPtr<CLImage> get_lsc_table ();
    virtual float* get_lsc_gray_threshold ();

private:
    SmartPtr<CLImage> &get_input_image (NV12PlaneIdx index) {
        XCAM_ASSERT (index < NV12PlaneMax);
        return _input [index];
    }
    SmartPtr<CLImage> &get_output_image (NV12PlaneIdx index) {
        XCAM_ASSERT (index < NV12PlaneMax);
        return _output [index];
    }

    SmartPtr<CLImage> create_cl_image (
        uint32_t width, uint32_t height, cl_channel_order order, cl_channel_type type);
    XCamReturn generate_fisheye_table (
        uint32_t fisheye_width, uint32_t fisheye_height, const FisheyeInfo &fisheye_info);

    void ensure_lsc_params ();
    XCamReturn generate_lsc_table (
        uint32_t fisheye_width, uint32_t fisheye_height, FisheyeInfo &fisheye_info);


    XCAM_DEAD_COPY (CLFisheyeHandler);

private:
    uint32_t                         _output_width;
    uint32_t                         _output_height;
    float                            _range_longitude;
    float                            _range_latitude;
    FisheyeInfo                    _fisheye_info;
    float                            _map_factor;
    bool                             _use_map;
    uint32_t                         _need_lsc;
    uint32_t                         _need_scale;
    uint32_t                         _lsc_array_size;
    float                            _gray_threshold[2];  // [min_gray_threshold, max_gray_threshold]
    float                            *_lsc_array;
    float                            _stable_y_start;

    PointFloat2                      _left_scale_factor;
    PointFloat2                      _right_scale_factor;

    BowlDataConfig                   _bowl_data_config;

    IntrinsicParameter               _intrinsic_param;
    ExtrinsicParameter               _extrinsic_param;

    SurroundMode                     _surround_mode;

    SmartPtr<CLImage>                _geo_table;
    SmartPtr<CLImage>                _lsc_table;
    SmartPtr<CLImage>                _input[NV12PlaneMax];
    SmartPtr<CLImage>                _output[NV12PlaneMax];
};

SmartPtr<CLImageHandler>
create_fisheye_handler (const SmartPtr<CLContext> &context, SurroundMode surround_mode = SphereView, bool use_map = false, bool need_lsc = false, bool need_scale = false);

}

#endif //XCAM_CL_FISHEYE_HANDLER_H

