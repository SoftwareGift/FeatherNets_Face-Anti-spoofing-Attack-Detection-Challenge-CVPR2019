/*
 * cl_va_memory.cpp - CL va memory
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

#include "cl_va_memory.h"
#include "cl_image_bo_buffer.h"

namespace XCam {

CLVaBuffer::CLVaBuffer (
    const SmartPtr<CLIntelContext> &context,
    SmartPtr<DrmBoBuffer> &bo)
    : CLBuffer (context)
    , _bo (bo)
{
    init_va_buffer (context, bo);
}

bool
CLVaBuffer::init_va_buffer (const SmartPtr<CLIntelContext> &context, SmartPtr<DrmBoBuffer> &bo)
{
    cl_mem mem_id = NULL;
    uint32_t bo_name = 0;
    cl_import_buffer_info_intel import_buffer_info;

    xcam_mem_clear (import_buffer_info);
    import_buffer_info.fd = bo->get_fd ();
    import_buffer_info.size = bo->get_size ();
    if (import_buffer_info.fd != -1) {
        mem_id = context->import_dma_buffer (import_buffer_info);
    }

    if (mem_id == NULL) {
        drm_intel_bo_flink (bo->get_bo (), &bo_name);
        mem_id = context->create_va_buffer (bo_name);
        if (mem_id == NULL) {
            XCAM_LOG_WARNING ("CLVaBuffer create va buffer failed");
            return false;
        }
    }

    set_mem_id (mem_id);
    return true;
}

CLVaImage::CLVaImage (
    const SmartPtr<CLIntelContext> &context,
    SmartPtr<DrmBoBuffer> &bo,
    uint32_t offset,
    bool single_plane)
    : CLImage (context)
    , _bo (bo)
{
    CLImageDesc cl_desc;

    const VideoBufferInfo & video_info = bo->get_video_info ();
    if (!video_info_2_cl_image_desc (video_info, cl_desc)) {
        XCAM_LOG_WARNING ("CLVaImage create va image failed on default videoinfo");
        return;
    }
    if (single_plane) {
        cl_desc.array_size = 0;
        cl_desc.slice_pitch = 0;
    } else if (!merge_multi_plane (video_info, cl_desc)) {
        XCAM_LOG_WARNING ("CLVaImage create va image failed on merging planes");
        return;
    }

    init_va_image (context, bo, cl_desc, offset);
}

CLVaImage::CLVaImage (
    const SmartPtr<CLIntelContext> &context,
    SmartPtr<DrmBoBuffer> &bo,
    const CLImageDesc &image_info,
    uint32_t offset)
    : CLImage (context)
    , _bo (bo)
{
    init_va_image (context, bo, image_info, offset);
}

bool
CLVaImage::merge_multi_plane (
    const VideoBufferInfo &video_info,
    CLImageDesc &cl_desc)
{
    if (cl_desc.array_size <= 1)
        return true;

    switch (video_info.format) {
    case V4L2_PIX_FMT_NV12:
        cl_desc.height = video_info.aligned_height + video_info.height / 2;
        break;

    case XCAM_PIX_FMT_RGB48_planar:
    case XCAM_PIX_FMT_RGB24_planar:
        cl_desc.height = video_info.aligned_height * 3;
        break;

    case XCAM_PIX_FMT_SGRBG16_planar:
    case XCAM_PIX_FMT_SGRBG8_planar:
        cl_desc.height = video_info.aligned_height * 4;
        break;

    default:
        XCAM_LOG_WARNING ("CLVaImage unknown format(%s) plane change", xcam_fourcc_to_string(video_info.format));
        return false;
    }
    cl_desc.array_size = 0;
    cl_desc.slice_pitch = 0;
    return true;
}

bool
CLVaImage::init_va_image (
    const SmartPtr<CLIntelContext> &context, SmartPtr<DrmBoBuffer> &bo,
    const CLImageDesc &cl_desc, uint32_t offset)
{

    uint32_t bo_name = 0;
    cl_mem mem_id = 0;
    bool need_create = true;
    cl_libva_image va_image_info;
    cl_import_image_info_intel import_image_info;

    xcam_mem_clear (va_image_info);
    xcam_mem_clear (import_image_info);
    import_image_info.offset = va_image_info.offset = offset;
    import_image_info.width = va_image_info.width = cl_desc.width;
    import_image_info.height = va_image_info.height = cl_desc.height;
    import_image_info.fmt = va_image_info.fmt = cl_desc.format;
    import_image_info.row_pitch = va_image_info.row_pitch = cl_desc.row_pitch;
    import_image_info.size = cl_desc.size;
    import_image_info.type = CL_MEM_OBJECT_IMAGE2D;

    XCAM_ASSERT (bo.ptr ());

    SmartPtr<CLImageBoBuffer> cl_image_buffer = bo.dynamic_cast_ptr<CLImageBoBuffer> ();
    if (cl_image_buffer.ptr ()) {
        SmartPtr<CLImage> cl_image_data = cl_image_buffer->get_cl_image ();
        XCAM_ASSERT (cl_image_data.ptr ());
        CLImageDesc old_desc = cl_image_data->get_image_desc ();
        if (cl_desc == old_desc) {
            need_create = false;
            mem_id = cl_image_data->get_mem_id ();
        }
    }

    if (need_create) {
        import_image_info.fd = bo->get_fd();
        if (import_image_info.fd != -1)
            mem_id = context->import_dma_image (import_image_info);

        if (mem_id == NULL) {
            if (drm_intel_bo_flink (bo->get_bo (), &bo_name) == 0) {
                va_image_info.bo_name = bo_name;
                mem_id = context->create_va_image (va_image_info);
            }
            if (mem_id == NULL) {
                XCAM_LOG_WARNING ("create va image failed");
                return false;
            }
        }
    } else {
        va_image_info.bo_name = uint32_t(-1);
    }

    set_mem_id (mem_id, need_create);
    init_desc_by_image ();
    _va_image_info = va_image_info;
    return true;
}

};
