/*
 * cl_geo_map_handler.h - CL geometry map handler
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

#ifndef XCAM_CL_GEO_MAP_HANDLER_H
#define XCAM_CL_GEO_MAP_HANDLER_H

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>

namespace XCam {

struct GeoPos {
    double x;
    double y;

    GeoPos () : x(0), y(0) {}
};

class CLGeoMapKernel;
class GeoKernelParamCallback
{
    friend class CLGeoMapKernel;

public:
    GeoKernelParamCallback () {}
    virtual ~GeoKernelParamCallback () {}

protected:
    virtual SmartPtr<CLImage> get_geo_input_image (NV12PlaneIdx index) = 0;
    virtual SmartPtr<CLImage> get_geo_output_image (NV12PlaneIdx index) = 0;
    virtual SmartPtr<CLImage> get_geo_map_table () = 0;
    virtual void get_geo_equivalent_out_size (float &width, float &height) = 0;
    virtual void get_geo_pixel_out_size (float &width, float &height) = 0;

    virtual SmartPtr<CLImage> get_lsc_table () = 0;
    virtual float* get_lsc_gray_threshold() = 0;

    virtual PointFloat2 get_left_scale_factor () = 0;
    virtual PointFloat2 get_right_scale_factor () = 0;

    virtual float get_stable_y_start () = 0;
private:
    XCAM_DEAD_COPY (GeoKernelParamCallback);
};

class CLGeoMapHandler;
class CLGeoMapKernel
    : public CLImageKernel
{
public:
    explicit CLGeoMapKernel (
        const SmartPtr<CLContext> &context,
        const SmartPtr<GeoKernelParamCallback> handler,
        bool need_lsc, bool need_scale);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    SmartPtr<GeoKernelParamCallback>   _handler;
    bool                               _need_lsc;
    bool                               _need_scale;
};

class CLGeoMapHandler
    : public CLImageHandler
    , public GeoKernelParamCallback
{
public:
    explicit CLGeoMapHandler (const SmartPtr<CLContext> &context);
    void set_output_size (uint32_t width, uint32_t height) {
        _output_width = width;
        _output_height = height;
    }
    void get_output_size (uint32_t &width, uint32_t &height) const {
        width = _output_width;
        height = _output_height;
    }

    bool set_map_data (GeoPos *data, uint32_t width, uint32_t height);
    bool set_map_uint (float uint_x, float uint_y);
    void get_map_uint (float &uint_x, float &uint_y) {
        uint_x = _uint_x;
        uint_y = _uint_y;
    }

protected:
    // derived from GeoKernelParamCallback
    virtual SmartPtr<CLImage> get_geo_input_image (NV12PlaneIdx index) {
        XCAM_ASSERT (index < NV12PlaneMax);
        return _input [index];
    }
    virtual SmartPtr<CLImage> get_geo_output_image (NV12PlaneIdx index) {
        XCAM_ASSERT (index < NV12PlaneMax);
        return _output [index];
    }
    virtual SmartPtr<CLImage> get_geo_map_table () {
        XCAM_ASSERT (_geo_image.ptr ());
        return _geo_image;
    }
    virtual void get_geo_equivalent_out_size (float &width, float &height);
    virtual void get_geo_pixel_out_size (float &width, float &height);

    virtual SmartPtr<CLImage> get_lsc_table () {
        XCAM_ASSERT (false && "CLGeoMapHandler::lsc table is not supported");
        return NULL;
    }
    virtual float* get_lsc_gray_threshold () {
        XCAM_ASSERT (false && "CLGeoMapHandler::lsc gray threshold is not supported");
        return NULL;
    }

    virtual PointFloat2 get_left_scale_factor () {
        XCAM_ASSERT (false && "CLGeoMapHandler::left_scale_factor is not supported");
        return PointFloat2 (0.0f, 0.0f);
    }

    virtual PointFloat2 get_right_scale_factor () {
        XCAM_ASSERT (false && "CLGeoMapHandler::right_scale_factor is not supported");
        return PointFloat2 (0.0f, 0.0f);
    }

    virtual float get_stable_y_start () {
        XCAM_ASSERT (false && "CLGeoMapHandler::get_stable_y_start is not supported");
        return 0.0f;
    }


protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

private:
    bool normalize_geo_map (uint32_t image_w, uint32_t image_h);
    bool check_geo_map_buf (uint32_t width, uint32_t height);

    XCAM_DEAD_COPY (CLGeoMapHandler);

private:
    uint32_t                         _output_width;
    uint32_t                         _output_height;
    uint32_t                         _map_width, _map_height;
    uint32_t                         _map_aligned_width;
    float                            _uint_x, _uint_y;
    SmartPtr<CLImage>                _input[NV12PlaneMax];
    SmartPtr<CLImage>                _output[NV12PlaneMax];
    SmartPtr<CLBuffer>               _geo_map;
    SmartPtr<CLImage>                _geo_image;
    bool                             _geo_map_normalized;
};

SmartPtr<CLImageKernel>
create_geo_map_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<GeoKernelParamCallback> param_cb, bool need_lsc, bool need_scale);

SmartPtr<CLImageHandler>
create_geo_map_handler (const SmartPtr<CLContext> &context, bool need_lsc = false, bool need_scale = false);

}

#endif //XCAM_CL_GEO_MAP_HANDLER_H
