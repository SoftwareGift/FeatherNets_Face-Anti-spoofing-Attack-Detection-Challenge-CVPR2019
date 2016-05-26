/*
 * cl_wire_frame_handler.h - CL wire frame handler
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_CL_WIRE_FRAME_H
#define XCAM_CL_WIRE_FRAME_H

#include "cl_image_handler.h"

#define XCAM_WIRE_FRAME_MAX_COUNT 160

namespace XCam {

typedef struct _CLWireFrame {
    uint32_t pos_x;
    uint32_t pos_y;
    uint32_t width;
    uint32_t height;
} CLWireFrame;

class CLWireFrameImageKernel
    : public CLImageKernel
{
public:
    explicit CLWireFrameImageKernel (SmartPtr<CLContext> &context, const char *name);
    bool set_wire_frame_config (const XCamFDResult *config, double scaler_factor);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    bool check_wire_frames_validity (uint32_t image_width, uint32_t image_height);
    uint32_t get_border_coordinates_num ();
    bool get_border_coordinates (uint32_t *coords);
    XCAM_DEAD_COPY (CLWireFrameImageKernel);

private:
    uint32_t                                 _wire_frames_num;
    uint32_t                                 _wire_frames_coords_num;
    uint32_t                                 *_wire_frames_coords;
    CLWireFrame                              _wire_frames [XCAM_WIRE_FRAME_MAX_COUNT];
    SmartPtr<CLImage>                        _image_out_uv;
    SmartPtr<CLBuffer>                       _wire_frames_coords_buf;
};

class CLWireFrameImageHandler
    : public CLImageHandler
{
public:
    explicit CLWireFrameImageHandler (const char *name);
    bool set_wire_frame_kernel (SmartPtr<CLWireFrameImageKernel> &kernel);
    bool set_wire_frame_config (const XCamFDResult *config, double scaler_factor = 1.0);

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLWireFrameImageHandler);
    SmartPtr<CLWireFrameImageKernel>         _wire_frame_kernel;
};

SmartPtr<CLImageHandler>
create_cl_wire_frame_image_handler (SmartPtr<CLContext> &context);

};

#endif // XCAM_CL_WIRE_FRAME_H
