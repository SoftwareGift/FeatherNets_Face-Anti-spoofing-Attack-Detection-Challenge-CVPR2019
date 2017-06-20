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
#include "cl_fisheye_handler.h"
#include "cl_blender.h"
#if HAVE_OPENCV
#include "cv_feature_match.h"
#endif

namespace XCam {

enum CLStitchResMode {
    CLStitchRes1080P,
    CLStitchRes4K
};

enum ImageIdx {
    ImageIdxMain,
    ImageIdxSecondary,
    ImageIdxCount,
};

struct CLFisheyeParams {
    SmartPtr<CLFisheyeHandler>  handler;
    SmartPtr<BufferPool>        pool;
    SmartPtr<DrmBoBuffer>       buf;

    uint32_t                    width;
    uint32_t                    height;

    CLFisheyeParams () : width (0), height (0) {}
};

struct ImageCropInfo {
    uint32_t left;
    uint32_t right;
    uint32_t top;
    uint32_t bottom;

    ImageCropInfo () : left (0), right (0), top (0), bottom (0) {}
};

struct CLStitchInfo {
    uint32_t merge_width[ImageIdxCount];

    ImageCropInfo crop[ImageIdxCount];
    CLFisheyeInfo fisheye_info[ImageIdxCount];

    CLStitchInfo () {
        xcam_mem_clear (merge_width);
    }
};

typedef struct {
    Rect left;
    Rect right;
} ImageMergeInfo;

class CLImage360Stitch;
class CLBlenderGlobalScaleKernel
    : public CLBlenderScaleKernel
{
public:
    explicit CLBlenderGlobalScaleKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLImage360Stitch> &stitch, bool is_uv);

protected:
    virtual SmartPtr<CLImage> get_input_image ();
    virtual SmartPtr<CLImage> get_output_image ();
    virtual bool get_output_info (uint32_t &out_width, uint32_t &out_height, int &out_offset_x);

private:
    SmartPtr<CLImage360Stitch>  _stitch;
};

class CLImage360Stitch
    : public CLMultiImageHandler
{
public:
    explicit CLImage360Stitch (
        const SmartPtr<CLContext> &context, CLBlenderScaleMode scale_mode, CLStitchResMode res_mode);

    bool set_stitch_info (CLStitchInfo stitch_info);
    CLStitchInfo get_stitch_info ();
    void set_output_size (uint32_t width, uint32_t height) {
        _output_width = width; //XCAM_ALIGN_UP (width, XCAM_BLENDER_ALIGNED_WIDTH);
        _output_height = height;
    }

    bool set_fisheye_handler (SmartPtr<CLFisheyeHandler> fisheye, int index);
    bool set_left_blender (SmartPtr<CLBlender> blender);
    bool set_right_blender (SmartPtr<CLBlender> blender);

    bool set_image_overlap (const int idx, const Rect &overlap0, const Rect &overlap1);
    const Rect &get_image_overlap (ImageIdx image, int num) {
        XCAM_ASSERT (image < ImageIdxCount && num < 2);
        return _overlaps[image][num];
    }

    SmartPtr<DrmBoBuffer> &get_global_scale_input () {
        return _scale_global_input;
    }
    SmartPtr<DrmBoBuffer> &get_global_scale_output () {
        return _scale_global_output;
    }

    void set_feature_match_ocl (bool use_ocl);
#if HAVE_OPENCV
    void set_feature_match_config (CVFMConfig config);
    CVFMConfig get_feature_match_config ();
#endif

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (const VideoBufferInfo &input, VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<DrmBoBuffer> &output);

    XCamReturn ensure_fisheye_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    XCamReturn prepare_local_scale_blender_parameters (
        SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output);
    XCamReturn prepare_global_scale_blender_parameters (
        SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output);

    bool create_buffer_pool (SmartPtr<BufferPool> &buf_pool, uint32_t width, uint32_t height);
    XCamReturn reset_buffer_info (SmartPtr<DrmBoBuffer> &input);

    virtual XCamReturn sub_handler_execute_done (SmartPtr<CLImageHandler> &handler);

    void calc_fisheye_initial_info (SmartPtr<DrmBoBuffer> &output);
    void update_image_overlap ();

private:
    XCAM_DEAD_COPY (CLImage360Stitch);

private:
    SmartPtr<CLContext>         _context;
    CLFisheyeParams             _fisheye[ImageIdxCount];
    SmartPtr<CLBlender>         _left_blender;
    SmartPtr<CLBlender>         _right_blender;
#if HAVE_OPENCV
    SmartPtr<CVFeatureMatch>    _feature_match;
#endif

    uint32_t                    _output_width;
    uint32_t                    _output_height;
    ImageMergeInfo              _img_merge_info[ImageIdxCount];
    Rect                        _overlaps[ImageIdxCount][2];   // 2=>Overlap0 and overlap1

    CLBlenderScaleMode          _scale_mode;
    SmartPtr<BufferPool>        _scale_buf_pool;
    SmartPtr<DrmBoBuffer>       _scale_global_input;
    SmartPtr<DrmBoBuffer>       _scale_global_output;

    CLStitchResMode             _res_mode;

    bool                        _is_stitch_inited;
    CLStitchInfo                _stitch_info;
};

SmartPtr<CLImageHandler>
create_image_360_stitch (
    const SmartPtr<CLContext> &context,
    bool need_seam = false,
    CLBlenderScaleMode scale_mode = CLBlenderScaleLocal,
    bool fisheye_map = false,
    CLStitchResMode res_mode = CLStitchRes1080P);

}

#endif //XCAM_CL_IMAGE_360_STITCH_H
