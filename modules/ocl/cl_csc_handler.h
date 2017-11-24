/*
 * cl_csc_handler.h - CL csc handler
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
 * Author: wangfei <feix.w.wang@intel.com>
 */

#ifndef XCAM_CL_CSC_HANLDER_H
#define XCAM_CL_CSC_HANLDER_H

#include <xcam_std.h>
#include <base/xcam_3a_result.h>
#include <ocl/cl_image_handler.h>

namespace XCam {

enum CLCscType {
    CL_CSC_TYPE_RGBATONV12,
    CL_CSC_TYPE_RGBATOLAB,
    CL_CSC_TYPE_RGBA64TORGBA,
    CL_CSC_TYPE_YUYVTORGBA,
    CL_CSC_TYPE_NV12TORGBA,
    CL_CSC_TYPE_MAX,
};

class CLCscImageKernel
    : public CLImageKernel
{
public:
    explicit CLCscImageKernel (const SmartPtr<CLContext> &context, CLCscType type);

private:
    CLCscType               _kernel_csc_type;
};

class CLCscImageHandler
    : public CLImageHandler
{
public:
    explicit CLCscImageHandler (const SmartPtr<CLContext> &context, const char *name, CLCscType type);
    bool set_csc_kernel (SmartPtr<CLCscImageKernel> &kernel);
    bool set_matrix (const XCam3aResultColorMatrix &matrix);
    bool set_output_format (uint32_t fourcc);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLCscImageHandler);

private:
    float                       _rgbtoyuv_matrix[XCAM_COLOR_MATRIX_SIZE];
    uint32_t                    _output_format;
    CLCscType                   _csc_type;
    SmartPtr<CLCscImageKernel>  _csc_kernel;
};

SmartPtr<CLImageHandler>
create_cl_csc_image_handler (const SmartPtr<CLContext> &context, CLCscType type);

};

#endif //XCAM_CL_CSC_HANLDER_H
