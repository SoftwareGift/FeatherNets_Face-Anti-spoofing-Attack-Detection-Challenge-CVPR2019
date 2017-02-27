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

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

enum CLGeoPlaneIdx {
    CLGeoPlaneY = 0,
    CLGeoPlaneUV,
    CLGeoPlaneMax,
};

struct GeoPos {
    double x;
    double y;

    GeoPos () : x(0), y(0) {}
};

class CLGeoMapHandler;
class CLGeoMapKernel
    : public CLImageKernel
{
public:
    explicit CLGeoMapKernel (SmartPtr<CLContext> &context, SmartPtr<CLGeoMapHandler> &handler);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    SmartPtr<CLGeoMapHandler>   _handler;
    float                       _geo_scale_size[2]; //width/height
    float                       _out_size[2]; //width/height
};

class CLGeoMapHandler
    : public CLImageHandler
{
public:
    explicit CLGeoMapHandler ();
    void set_output_size (uint32_t width, uint32_t height) {
        _output_width = width;
        _output_height = height;
    }
    uint32_t get_output_width () const {
        return _output_width;
    }
    uint32_t get_output_height () const {
        return _output_height;
    }

    bool set_map_data (GeoPos *data, uint32_t width, uint32_t height);
    void set_map_uint (float uint_x, float uint_y) {
        _uint_x = uint_x;
        _uint_y = uint_y;
    }

    SmartPtr<CLImage> &get_input_image (CLGeoPlaneIdx index) {
        XCAM_ASSERT (index < CLGeoPlaneMax);
        return _input [index];
    }
    SmartPtr<CLImage> &get_output_image (CLGeoPlaneIdx index) {
        XCAM_ASSERT (index < CLGeoPlaneMax);
        return _output [index];
    }
    SmartPtr<CLImage> &get_geo_map_image () {
        return _geo_image;
    }
    void get_geo_equivalent_out_size (float &width, float &height) const;

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<DrmBoBuffer> &output);

private:
    XCamReturn normalize_geo_map (uint32_t image_w, uint32_t image_h);

    XCAM_DEAD_COPY (CLGeoMapHandler);

private:
    uint32_t                         _output_width;
    uint32_t                         _output_height;
    uint32_t                         _map_width;
    uint32_t                         _map_height;
    float                            _uint_x, _uint_y;
    SmartPtr<CLImage>                _input[CLGeoPlaneMax];
    SmartPtr<CLImage>                _output[CLGeoPlaneMax];
    SmartPtr<CLBuffer>               _geo_map;
    SmartPtr<CLImage>                _geo_image;
    bool                             _geo_map_normalized;
};

SmartPtr<CLImageHandler>
create_geo_map_handler (SmartPtr<CLContext> &context);

}

#endif //XCAM_CL_GEO_MAP_HANDLER_H