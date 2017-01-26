/*
 * cl_biyuv_handler.h - CL Biyuv handler
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: Juan Zhao <juan.j.zhao@intel.com>
 *             Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CL_BILATERAL_HANLDER_H
#define XCAM_CL_BILATERAL_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

class CLBilateralKernel
    : public CLImageKernel
{
public:
    explicit CLBilateralKernel (SmartPtr<CLContext> &context, bool is_rgb);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLBilateralKernel);
    float        _sigma_r;
    uint32_t     _imw;
    uint32_t     _imh;
    uint32_t     _vertical_offset;
    bool         _is_rgb;
};

class CLBilateralImageHandler
    : public CLImageHandler
{
public:
    explicit CLBilateralImageHandler (const char *name);

private:
    XCAM_DEAD_COPY (CLBilateralImageHandler);

};

SmartPtr<CLImageHandler>
create_cl_bilateral_image_handler (SmartPtr<CLContext> &context, bool is_rgb);

};

#endif //XCAM_CL_BILATERAL_HANLDER_H