/*
 * cl_pyramid_blender.h - CL pyramid blender
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

#ifndef XCAM_CL_PYRAMID_BLENDER_H
#define XCAM_CL_PYRAMID_BLENDER_H

#include "xcam_utils.h"
#include "cl_blender.h"

#define CL_PYRAMID_ENABLE_DUMP 0

#define XCAM_CL_PYRAMID_MAX_LEVEL  4

namespace XCam {

class CLPyramidBlender;

enum {
    BlendImageIndex = 0,
    ReconstructImageIndex,
    BlendImageCount
};

struct PyramidLayer {
    uint32_t                 blend_width; // blend, gauss, and lap
    uint32_t                 blend_height;
    SmartPtr<CLImage>        gauss_image[CLBlenderPlaneMax][XCAM_CL_BLENDER_IMAGE_NUM];
    int32_t                  gauss_offset_x[CLBlenderPlaneMax][XCAM_CL_BLENDER_IMAGE_NUM]; // aligned with XCAM_BLENDER_ALIGNED_WIDTH
    SmartPtr<CLImage>        lap_image[CLBlenderPlaneMax][XCAM_CL_BLENDER_IMAGE_NUM];
    int32_t                  lap_offset_x[CLBlenderPlaneMax][XCAM_CL_BLENDER_IMAGE_NUM]; // aligned with XCAM_BLENDER_ALIGNED_WIDTH
    SmartPtr<CLImage>        blend_image[CLBlenderPlaneMax][BlendImageCount]; // 0 blend-image, 1 reconstruct image
    uint32_t                 mask_width[CLBlenderPlaneMax];
    SmartPtr<CLBuffer>       blend_mask[CLBlenderPlaneMax]; // sizeof(float) * mask_width

#if CL_PYRAMID_ENABLE_DUMP
    SmartPtr<CLImage>        dump_gauss_resize[CLBlenderPlaneMax];
    SmartPtr<CLImage>        dump_original[CLBlenderPlaneMax][BlendImageCount];
    SmartPtr<CLImage>        dump_final[CLBlenderPlaneMax];
#endif

    PyramidLayer ();
    void bind_buf_to_layer0 (
        SmartPtr<CLContext> context,
        SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output,
        const Rect &merge0_rect, const Rect &merge1_rect, bool need_uv);
    void init_layer0 (SmartPtr<CLContext> context, bool last_layer, bool need_uv, int mask_radius, float mask_sigma);
    void build_cl_images (SmartPtr<CLContext> context, bool need_lap, bool need_uv);
    bool copy_mask_from_y_to_uv (SmartPtr<CLContext> &context);
};

class CLLinearBlenderKernel;

class CLPyramidBlendKernel;

class CLPyramidBlender
    : public CLBlender
{
    friend class CLPyramidBlendKernel;

public:
    explicit CLPyramidBlender (const char *name, int layers, bool need_uv);
    ~CLPyramidBlender ();

    //void set_blend_kernel (SmartPtr<CLLinearBlenderKernel> kernel, int index);
    SmartPtr<CLImage> get_gauss_image (uint32_t layer, uint32_t buf_index, bool is_uv);
    SmartPtr<CLImage> get_lap_image (uint32_t layer, uint32_t buf_index, bool is_uv);
    SmartPtr<CLImage> get_blend_image (uint32_t layer, bool is_uv);
    SmartPtr<CLImage> get_reconstruct_image (uint32_t layer, bool is_uv);
    SmartPtr<CLBuffer> get_blend_mask (uint32_t layer, bool is_uv);
    const PyramidLayer &get_pyramid_layer (uint32_t layer) const;

protected:
    // from CLImageHandler
    virtual XCamReturn execute_done (SmartPtr<DrmBoBuffer> &output);

    // from CLBlender
    virtual XCamReturn allocate_cl_buffers (
        SmartPtr<CLContext> context, SmartPtr<DrmBoBuffer> &input0,
        SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output);

private:
    void last_layer_buffer_redirect ();

    void dump_layer_mask (uint32_t layer, bool is_uv);
    void dump_buffers ();

    XCAM_DEAD_COPY (CLPyramidBlender);

private:
    uint32_t                         _layers;
    PyramidLayer                     _pyramid_layers[XCAM_CL_PYRAMID_MAX_LEVEL];
};

class CLPyramidBlendKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidBlendKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
private:
    SmartPtr<CLImage> get_input_0 () {
        return _blender->get_lap_image (_layer, 0, _is_uv);
    }
    SmartPtr<CLImage> get_input_1 () {
        return _blender->get_lap_image (_layer, 1, _is_uv);
    }
    SmartPtr<CLImage> get_ouput () {
        return _blender->get_blend_image (_layer, _is_uv);
    }
    SmartPtr<CLBuffer> get_blend_mask () {
        return _blender->get_blend_mask (_layer, _is_uv);
    }
private:
    XCAM_DEAD_COPY (CLPyramidBlendKernel);

private:
    SmartPtr<CLPyramidBlender>     _blender;
    uint32_t                       _layer;
    bool                           _is_uv;

};

class CLPyramidTransformKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidTransformKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, uint32_t buf_index, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    SmartPtr<CLImage> get_input_gauss () {
        return _blender->get_gauss_image (_layer, _buf_index, _is_uv);
    }
    int32_t get_input_gauss_offset_x ();
    SmartPtr<CLImage> get_output_gauss () {
        // need reset format
        return _blender->get_gauss_image (_layer + 1, _buf_index, _is_uv);
    }


    XCAM_DEAD_COPY (CLPyramidTransformKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    uint32_t                           _layer;
    uint32_t                           _buf_index;
    bool                               _is_uv;
    SmartPtr<CLImage>                  _output_gauss;
    int                                _gauss_offset_x;
};

class CLPyramidLapKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidLapKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, uint32_t buf_index, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    SmartPtr<CLImage> get_current_gauss () {
        return _blender->get_gauss_image (_layer, _buf_index, _is_uv);
    }
    SmartPtr<CLImage> get_next_gauss () {
        return _blender->get_gauss_image (_layer + 1, _buf_index, _is_uv);
    }
    int32_t get_cur_gauss_offset_x ();
    int32_t get_output_lap_offset_x ();

    SmartPtr<CLImage> get_output_lap () {
        return _blender->get_lap_image (_layer, _buf_index, _is_uv);
    }

    XCAM_DEAD_COPY (CLPyramidLapKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    uint32_t                           _layer;
    uint32_t                           _buf_index;
    bool                               _is_uv;
    SmartPtr<CLImage>                  _next_gauss;
    int                                _cur_gauss_offset_x;
    int                                _lap_offset_x;
    //float                              _ratio_x, _ratio_y;
    float                              _sampler_offset_x, _sampler_offset_y;
    float                              _out_width, _out_height;
};


class CLPyramidReconstructKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidReconstructKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    SmartPtr<CLImage>  get_input_reconstruct () {
        return _blender->get_reconstruct_image (_layer + 1, _is_uv);
    }
    SmartPtr<CLImage>  get_input_lap () {
        return _blender->get_blend_image (_layer, _is_uv);
    }
    SmartPtr<CLImage>  get_output_reconstruct () {
        return _blender->get_reconstruct_image (_layer, _is_uv);
    }

    int get_output_reconstrcut_offset_x ();


    XCAM_DEAD_COPY (CLPyramidReconstructKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    uint32_t                           _layer;
    bool                               _is_uv;
    SmartPtr<CLImage>                  _input_reconstruct;
    float                              _in_sampler_offset_x, _in_sampler_offset_y;
    float                              _out_reconstruct_width;
    float                              _out_reconstruct_height;
    int                                _out_reconstruct_offset_x;
};

class CLPyramidCopyKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidCopyKernel (
        SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t buf_index, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    SmartPtr<CLImage>  get_input () {
        return _blender->get_gauss_image (0, _buf_index, _is_uv);
    }
    SmartPtr<CLImage>  get_output () {
        return _blender->get_reconstruct_image (0, _is_uv);
    }

    XCAM_DEAD_COPY (CLPyramidCopyKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    bool                               _is_uv;
    int                                _buf_index;

    // parameters
    int                                _in_offset_x;
    int                                _out_offset_x;
    int                                _max_g_x;
    int                                _max_g_y;
    SmartPtr<CLImage>                  _from;
    SmartPtr<CLImage>                  _to;
};

};

#endif //XCAM_CL_PYRAMID_BLENDER_H

