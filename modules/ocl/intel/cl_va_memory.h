/*
 * cl_va_memory.h - CL va memory
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

#ifndef XCAM_CL_VA_MEMORY_H
#define XCAM_CL_VA_MEMORY_H

#include "ocl/cl_memory.h"
#include "ocl/intel/cl_intel_context.h"
#include "drm_bo_buffer.h"

namespace XCam {

class CLVaBuffer
    : public CLBuffer
{
public:
    explicit CLVaBuffer (
        const SmartPtr<CLIntelContext> &context,
        SmartPtr<DrmBoBuffer> &bo);

private:
    bool init_va_buffer (const SmartPtr<CLIntelContext> &context, SmartPtr<DrmBoBuffer> &bo);

    XCAM_DEAD_COPY (CLVaBuffer);

private:
    SmartPtr<DrmBoBuffer>   _bo;
};

class CLVaImage
    : public CLImage
{
public:
    explicit CLVaImage (
        const SmartPtr<CLIntelContext> &context,
        SmartPtr<DrmBoBuffer> &bo,
        uint32_t offset = 0,
        bool single_plane = false);
    explicit CLVaImage (
        const SmartPtr<CLIntelContext> &context,
        SmartPtr<DrmBoBuffer> &bo,
        const CLImageDesc &image_info,
        uint32_t offset = 0);
    ~CLVaImage () {}

private:
    bool init_va_image (
        const SmartPtr<CLIntelContext> &context, SmartPtr<DrmBoBuffer> &bo,
        const CLImageDesc &cl_desc, uint32_t offset);
    bool merge_multi_plane (
        const VideoBufferInfo &video_info,
        CLImageDesc &cl_desc);

    XCAM_DEAD_COPY (CLVaImage);

private:
    SmartPtr<DrmBoBuffer>   _bo;
    cl_libva_image          _va_image_info;
};

};
#endif //
