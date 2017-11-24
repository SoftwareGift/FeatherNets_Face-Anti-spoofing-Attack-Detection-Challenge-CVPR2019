/*
 * cl_tonemapping_handler.h - CL tonemapping handler
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: Wu Junkai <junkai.wu@intel.com>
 */

#ifndef XCAM_CL_TONEMAPPING_HANLDER_H
#define XCAM_CL_TONEMAPPING_HANLDER_H

#include <xcam_std.h>
#include <x3a_stats_pool.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_bayer_basic_handler.h>

namespace XCam {

class CLTonemappingImageKernel
    : public CLImageKernel
{
public:
    explicit CLTonemappingImageKernel (
        const SmartPtr<CLContext> &context, const char *name);
};

class CLTonemappingImageHandler
    : public CLImageHandler
{
public:
    explicit CLTonemappingImageHandler (const SmartPtr<CLContext> &context, const char *name);
    bool set_tonemapping_kernel(SmartPtr<CLTonemappingImageKernel> &kernel);
    bool set_wb_config (const XCam3aResultWhiteBalance &wb);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input, VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (
        SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLTonemappingImageHandler);
    SmartPtr<CLTonemappingImageKernel>   _tonemapping_kernel;
    int32_t                              _output_format;
    CLWBConfig                           _wb_config;
};

SmartPtr<CLImageHandler>
create_cl_tonemapping_image_handler (const SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_TONEMAPPING_HANLDER_H
