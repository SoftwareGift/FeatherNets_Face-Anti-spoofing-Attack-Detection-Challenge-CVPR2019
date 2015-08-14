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
 * Author: Yao Wang <yao.y.wang@intel.com>
 */

#ifndef XCAM_CL_TONEMAPPING_HANLDER_H
#define XCAM_CL_TONEMAPPING_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

class CLTonemappingImageKernel
    : public CLImageKernel
{
public:
    explicit CLTonemappingImageKernel (SmartPtr<CLContext> &context,
                                       const char *name);
    void set_initial_color_bits(uint32_t color_bits);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLTonemappingImageKernel);
    uint32_t _initial_color_bits;// color bits from ISP
};

class CLTonemappingImageHandler
    : public CLImageHandler
{
public:
    explicit CLTonemappingImageHandler (const char *name);
    bool set_tonemapping_kernel(SmartPtr<CLTonemappingImageKernel> &kernel);
    void set_initial_color_bits(uint32_t color_bits);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCAM_DEAD_COPY (CLTonemappingImageHandler);
    SmartPtr<CLTonemappingImageKernel>  _tonemapping_kernel;
    int32_t  _output_format;
};

SmartPtr<CLImageHandler>
create_cl_tonemapping_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_TONEMAPPING_HANLDER_H
