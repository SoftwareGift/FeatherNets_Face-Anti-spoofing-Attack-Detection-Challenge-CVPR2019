/*
 * cl_demo_handler.h - CL demo handler
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CL_DEMO_HANLDER_H
#define XCAM_CL_DEMO_HANLDER_H

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>

namespace XCam {

class CLDemoImageHandler
    : public CLImageHandler
{
public:
    explicit CLDemoImageHandler (const SmartPtr<CLContext> &context);
    void set_copy_kernel (SmartPtr<CLImageKernel> &kernel) {
        _copy_kernel = kernel;
        add_kernel (kernel);
    }

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    SmartPtr<CLImageKernel>   _copy_kernel;
};

SmartPtr<CLImageHandler>
create_cl_demo_image_handler (const SmartPtr<CLContext> &context);

SmartPtr<CLImageHandler>
create_cl_binary_demo_image_handler (const SmartPtr<CLContext> &context, const uint8_t *binary, size_t size);

};

#endif //XCAM_CL_DEMO_HANLDER_H
