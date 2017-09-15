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
#include "ocl/cl_image_handler.h"

namespace XCam {

class CLBilateralKernel
    : public CLImageKernel
{
public:
    explicit CLBilateralKernel (const SmartPtr<CLContext> &context);
};

class CLBilateralImageHandler
    : public CLImageHandler
{
public:
    explicit CLBilateralImageHandler (
        const SmartPtr<CLContext> &context, const char *name, bool is_rgb);
    void set_bi_kernel (SmartPtr<CLBilateralKernel> &kernel);

protected:
    virtual XCamReturn prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    SmartPtr<CLBilateralKernel>  _kernel;
    bool                         _is_rgb;
};

SmartPtr<CLImageHandler>
create_cl_bilateral_image_handler (const SmartPtr<CLContext> &context, bool is_rgb);

};

#endif //XCAM_CL_BILATERAL_HANLDER_H