/*
 * cl_intel_context.h - CL intel context
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

#ifndef XCAM_CL_INTEL_CONTEXT_H
#define XCAM_CL_INTEL_CONTEXT_H

#include <CL/cl_intel.h>
#include <ocl/cl_context.h>

namespace XCam {

class CLIntelContext
    : public CLContext
{
    friend class CLMemory;
    friend class CLDevice;
    friend class CLVaBuffer;
    friend class CLVaImage;

public:
    ~CLIntelContext () {}

private:
    explicit CLIntelContext (SmartPtr<CLDevice> &device);

    cl_mem create_va_buffer (uint32_t bo_name);
    cl_mem import_dma_buffer (const cl_import_buffer_info_intel &import_info);
    cl_mem create_va_image (const cl_libva_image &image_info);
    cl_mem import_dma_image (const cl_import_image_info_intel &image_info);

    int32_t export_mem_fd (cl_mem mem_id);

private:
    XCAM_DEAD_COPY (CLIntelContext);
};

};

#endif //XCAM_CL_CONTEXT_H
