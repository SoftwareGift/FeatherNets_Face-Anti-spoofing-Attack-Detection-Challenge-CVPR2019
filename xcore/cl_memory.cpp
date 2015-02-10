/*
 * cl_memory.cpp - CL memory
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

#include "cl_memory.h"
#include "drm_display.h"

namespace XCam {

CLMemory::CLMemory (SmartPtr<CLContext> &context)
    : _context (context)
    , _mem_id (NULL)
{
}

CLMemory::~CLMemory ()
{
    if (_mem_id) {
        _context->destroy_mem (_mem_id);
    }
}

CLVaImage::CLVaImage (
    SmartPtr<CLContext> &context,
    SmartPtr<DrmBoBuffer> &bo,
    const cl_libva_image *image_info)
    : CLMemory (context)
    , _bo (bo)
{
    uint32_t bo_name = 0;

    XCAM_ASSERT (context.ptr () && context->is_valid ());

    if (image_info) {
        _image_info = *image_info;
    } else {
        const VideoBufferInfo & video_info = bo->get_video_info ();
        xcam_mem_clear (&_image_info);
        _image_info.fmt.image_channel_order = CL_RGBA;
        _image_info.fmt.image_channel_data_type = CL_UNORM_INT8;
        _image_info.offset = 0;
        _image_info.width = video_info.width;
        _image_info.height = video_info.width;
        _image_info.row_pitch = video_info.strides[0];
    }

    if (drm_intel_bo_flink (bo->get_bo (), &bo_name) != 0) {
        XCAM_LOG_WARNING ("CLVaImage get bo flick failed");
    } else {
        _image_info.bo_name = bo_name;
        _mem_id = context->create_va_image (_image_info);
        if (_mem_id == NULL) {
            XCAM_LOG_WARNING ("create va image failed");
        }
    }
}

};
