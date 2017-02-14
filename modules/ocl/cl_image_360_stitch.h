/*
 * cl_image_360_stitch.h - CL Image 360 stitch
 *
 *  Copyright (c) 2016 Intel Corporation
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

#ifndef XCAM_CL_IMAGE_360_STITCH_H
#define XCAM_CL_IMAGE_360_STITCH_H

#include "xcam_utils.h"
#include "cl_multi_image_handler.h"
#include "cl_blender.h"

namespace XCam {

class CLImage360Stitch
    : public CLMultiImageHandler
{
public:
    enum ImageIdx {
        ImageIdxMain,
        ImageIdxSecondary,
        ImageIdxCount,
    };
public:
    explicit CLImage360Stitch (CLBlenderScaleMode scale_mode);

    void set_output_size (uint32_t width, uint32_t height) {
        _output_width = width; //XCAM_ALIGN_UP (width, XCAM_BLENDER_ALIGNED_WIDTH);
        _output_height = height;
    }
    bool set_left_blender (SmartPtr<CLBlender> blender);
    bool set_right_blender (SmartPtr<CLBlender> blender);

    bool set_image_overlap (const int idx, const Rect &overlap0, const Rect &overlap1);

    const Rect &get_image_overlap (ImageIdx image, int num) {
        XCAM_ASSERT (image < ImageIdxCount && num < 2);
        return _overlaps[image][num];
    }

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

    XCamReturn prepare_global_scale_blender_parameters (
        SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output);

    XCamReturn prepare_local_scale_blender_parameters (
        SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLImage360Stitch);

private:
    SmartPtr<CLBlender>    _left_blender;
    SmartPtr<CLBlender>    _right_blender;
    uint32_t               _output_width;
    uint32_t               _output_height;
    Rect                   _overlaps[ImageIdxCount][2];   // 2=>Overlap0 and overlap1
    CLBlenderScaleMode     _scale_mode;
};

SmartPtr<CLImageHandler>
create_image_360_stitch (
    SmartPtr<CLContext> &context, bool need_seam = false, CLBlenderScaleMode scale_mode = CLBlenderScaleLocal);

}

#endif //XCAM_CL_IMAGE_360_STITCH_H
