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

#include <xcam_std.h>
#include <ocl/cl_blender.h>

#define CL_PYRAMID_ENABLE_DUMP 0

#define XCAM_CL_PYRAMID_MAX_LEVEL  4

namespace XCam {

class CLPyramidBlender;

enum {
    BlendImageIndex = 0,
    ReconstructImageIndex,
    BlendImageCount
};

enum {
    CLSeamMaskTmp = 0,
    CLSeamMaskCoeff,
    CLSeamMaskCount
};

struct PyramidLayer {
    uint32_t                 blend_width; // blend, gauss, and lap
    uint32_t                 blend_height;
    SmartPtr<CLImage>        gauss_image[CLBlenderPlaneMax][XCAM_BLENDER_IMAGE_NUM];
    int32_t                  gauss_offset_x[CLBlenderPlaneMax][XCAM_BLENDER_IMAGE_NUM]; // aligned with XCAM_BLENDER_ALIGNED_WIDTH
    SmartPtr<CLImage>        lap_image[CLBlenderPlaneMax][XCAM_BLENDER_IMAGE_NUM];
    int32_t                  lap_offset_x[CLBlenderPlaneMax][XCAM_BLENDER_IMAGE_NUM]; // aligned with XCAM_BLENDER_ALIGNED_WIDTH
    SmartPtr<CLImage>        blend_image[CLBlenderPlaneMax][BlendImageCount]; // 0 blend-image, 1 reconstruct image
    uint32_t                 mask_width[CLBlenderPlaneMax];
    SmartPtr<CLBuffer>       blend_mask[CLBlenderPlaneMax]; // sizeof(float) * mask_width
    SmartPtr<CLImage>        seam_mask[CLSeamMaskCount];
    SmartPtr<CLImage>        scale_image[CLBlenderPlaneMax];

#if CL_PYRAMID_ENABLE_DUMP
    SmartPtr<CLImage>        dump_gauss_resize[CLBlenderPlaneMax];
    SmartPtr<CLImage>        dump_original[CLBlenderPlaneMax][BlendImageCount];
    SmartPtr<CLImage>        dump_final[CLBlenderPlaneMax];
#endif

    PyramidLayer ();
    void bind_buf_to_layer0 (
        SmartPtr<CLContext> context,
        SmartPtr<VideoBuffer> &input0, SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output,
        const Rect &merge0_rect, const Rect &merge1_rect, bool need_uv, CLBlenderScaleMode scale_mode);
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
    explicit CLPyramidBlender (
        const SmartPtr<CLContext> &context, const char *name,
        int layers, bool need_uv, bool need_seam, CLBlenderScaleMode scale_mode);
    ~CLPyramidBlender ();

    //void set_blend_kernel (SmartPtr<CLLinearBlenderKernel> kernel, int index);
    SmartPtr<CLImage> get_gauss_image (uint32_t layer, uint32_t buf_index, bool is_uv);
    SmartPtr<CLImage> get_lap_image (uint32_t layer, uint32_t buf_index, bool is_uv);
    SmartPtr<CLImage> get_blend_image (uint32_t layer, bool is_uv);
    SmartPtr<CLImage> get_reconstruct_image (uint32_t layer, bool is_uv);
    SmartPtr<CLImage> get_scale_image (bool is_uv);
    SmartPtr<CLBuffer> get_blend_mask (uint32_t layer, bool is_uv);
    SmartPtr<CLImage> get_seam_mask (uint32_t layer);
    const PyramidLayer &get_pyramid_layer (uint32_t layer) const;
    const SmartPtr<CLImage> &get_image_diff () const;
    void get_seam_info (uint32_t &width, uint32_t &height, uint32_t &stride) const;
    void get_seam_pos_info (uint32_t &offset_x, uint32_t &valid_width) const;
    SmartPtr<CLBuffer> &get_seam_pos_buf () {
        return _seam_pos_buf;
    }
    SmartPtr<CLBuffer> &get_seam_sum_buf () {
        return _seam_sum_buf;
    }
    uint32_t get_layers () const {
        return _layers;
    }
    XCamReturn fill_seam_mask ();

protected:
    // from CLImageHandler
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

    // from CLBlender
    virtual XCamReturn allocate_cl_buffers (
        SmartPtr<CLContext> context, SmartPtr<VideoBuffer> &input0,
        SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output);

private:
    XCamReturn init_seam_buffers (SmartPtr<CLContext> context);
    void last_layer_buffer_redirect ();

    void dump_layer_mask (uint32_t layer, bool is_uv);
    void dump_buffers ();

    XCAM_DEAD_COPY (CLPyramidBlender);

private:
    uint32_t                         _layers;
    PyramidLayer                     _pyramid_layers[XCAM_CL_PYRAMID_MAX_LEVEL];

    //calculate seam masks
    bool                             _need_seam;
    SmartPtr<CLImage>                _image_diff; // image difference in blending area, only Y
    uint32_t                         _seam_pos_stride;
    uint32_t                         _seam_width, _seam_height;
    uint32_t                         _seam_pos_offset_x, _seam_pos_valid_width;
    SmartPtr<CLBuffer>               _seam_pos_buf; // width = _seam_width; height = _seam_height;
    SmartPtr<CLBuffer>               _seam_sum_buf; // size = _seam_width
    bool                             _seam_mask_done;
    //SmartPtr<CLImage>                _seam_mask;
};

class CLPyramidBlendKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidBlendKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
        uint32_t layer, bool is_uv, bool need_seam);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);
private:
    SmartPtr<CLImage> get_input_0 () {
        return _blender->get_lap_image (_layer, 0, _is_uv);
    }
    SmartPtr<CLImage> get_input_1 () {
        return _blender->get_lap_image (_layer, 1, _is_uv);
    }
    SmartPtr<CLImage> get_output () {
        return _blender->get_blend_image (_layer, _is_uv);
    }
    SmartPtr<CLBuffer> get_blend_mask () {
        return _blender->get_blend_mask (_layer, _is_uv);
    }
    SmartPtr<CLImage> get_seam_mask () {
        return _blender->get_seam_mask (_layer);
    }
private:
    XCAM_DEAD_COPY (CLPyramidBlendKernel);

private:
    SmartPtr<CLPyramidBlender>     _blender;
    uint32_t                       _layer;
    bool                           _is_uv;
    bool                           _need_seam;

};

class CLPyramidTransformKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidTransformKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
        uint32_t layer, uint32_t buf_index, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

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
};

class CLSeamDiffKernel
    : public CLImageKernel
{
public:
    explicit CLSeamDiffKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    SmartPtr<CLPyramidBlender>         _blender;

};

class CLSeamDPKernel
    : public CLImageKernel
{
public:
    explicit CLSeamDPKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    int                                _seam_stride;
    int                                _seam_height;

};

class CLPyramidSeamMaskKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidSeamMaskKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
        uint32_t layer, bool scale, bool need_slm);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    int                                _layer;
    bool                               _need_scale;
    bool                               _need_slm;
};

class CLPyramidLapKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidLapKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
        uint32_t layer, uint32_t buf_index, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

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
};

class CLPyramidReconstructKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidReconstructKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
        uint32_t layer, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

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
};

class CLBlenderLocalScaleKernel
    : public CLBlenderScaleKernel
{
public:
    explicit CLBlenderLocalScaleKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, bool is_uv);

protected:
    virtual SmartPtr<CLImage> get_input_image ();
    virtual SmartPtr<CLImage> get_output_image ();

    virtual bool get_output_info (uint32_t &out_width, uint32_t &out_height, int &out_offset_x);

private:
    XCAM_DEAD_COPY (CLBlenderLocalScaleKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    SmartPtr<CLImage>                  _image_in;
};

class CLPyramidCopyKernel
    : public CLImageKernel
{
public:
    explicit CLPyramidCopyKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
        uint32_t buf_index, bool is_uv);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    SmartPtr<CLImage>  get_input () {
        return _blender->get_gauss_image (0, _buf_index, _is_uv);
    }
    SmartPtr<CLImage>  get_output () {
        if (_blender->get_scale_mode () == CLBlenderScaleLocal)
            return _blender->get_scale_image (_is_uv);
        else
            return _blender->get_reconstruct_image (0, _is_uv);
    }

    XCAM_DEAD_COPY (CLPyramidCopyKernel);

private:
    SmartPtr<CLPyramidBlender>         _blender;
    bool                               _is_uv;
    int                                _buf_index;

    // parameters
    int                                _max_g_x;
    int                                _max_g_y;
};

};

#endif //XCAM_CL_PYRAMID_BLENDER_H

