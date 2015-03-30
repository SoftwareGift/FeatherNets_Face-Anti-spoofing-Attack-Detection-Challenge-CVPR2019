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

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

enum CLCscType {
    CL_CSC_TYPE_NONE = 0,
    CL_CSC_TYPE_RGBATONV12,
    CL_CSC_TYPE_RGBATOLAB,
};

class CLCscImageKernel
    : public CLImageKernel
{
public:
    explicit CLCscImageKernel (SmartPtr<CLContext> &context, const char *name);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLCscImageKernel);

    uint32_t _vertical_offset;
};

class CLCscImageHandler
    : public CLImageHandler
{
public:
    explicit CLCscImageHandler (const char *name, CLCscType type);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCAM_DEAD_COPY (CLCscImageHandler);

private:
    uint32_t  _output_format;
    CLCscType _csc_type;
};

SmartPtr<CLImageHandler>
create_cl_csc_image_handler (SmartPtr<CLContext> &context, CLCscType type);

};

#endif //XCAM_CL_CSC_HANLDER_H
