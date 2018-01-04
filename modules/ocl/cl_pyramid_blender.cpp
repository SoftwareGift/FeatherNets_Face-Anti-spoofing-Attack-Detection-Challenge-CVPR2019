/*
 * cl_pyramid_blender.cpp - CL multi-band blender
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

#include "cl_pyramid_blender.h"
#include <algorithm>
#include "xcam_obj_debug.h"
#include "cl_device.h"
#include "cl_utils.h"

#if CL_PYRAMID_ENABLE_DUMP
#define BLENDER_PROFILING_START(name)  XCAM_STATIC_PROFILING_START(name)
#define BLENDER_PROFILING_END(name, times_of_print)  XCAM_STATIC_PROFILING_END(name, times_of_print)
#else
#define BLENDER_PROFILING_START(name)
#define BLENDER_PROFILING_END(name, times_of_print)
#endif

//#define SAMPLER_POSITION_OFFSET -0.25f
#define SAMPLER_POSITION_OFFSET 0.0f

#define SEAM_POS_TYPE int16_t
#define SEAM_SUM_TYPE float
#define SEAM_MASK_TYPE uint8_t

namespace XCam {

enum {
    KernelPyramidTransform   = 0,
    KernelPyramidReconstruct,
    KernelPyramidBlender,
    KernelPyramidScale,
    KernelPyramidCopy,
    KernelPyramidLap,
    KernelImageDiff,
    KernelSeamDP,
    KernelSeamMaskScale,
    KernelSeamMaskScaleSLM,
    KernelSeamBlender
};

static const XCamKernelInfo kernels_info [] = {
    {
        "kernel_gauss_scale_transform",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_gauss_lap_reconstruct",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_pyramid_blend",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_pyramid_scale",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_pyramid_copy",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_lap_transform",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_image_diff",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_seam_dp",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_mask_gauss_scale",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_mask_gauss_scale_slm",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_seam_mask_blend",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    }
};

static uint32_t
clamp(int32_t i, int32_t min, int32_t max)
{
    if (i < min)
        return min;
    if (i > max - 1)
        return max - 1;
    return i;
}

static float*
get_gauss_coeffs (int radius, float sigma)
{
    static int g_radius = 0;
    static float g_sigma = 0;
    static float g_table[512] = {0.0f};

    int i;
    int scale = radius * 2 + 1;
    float dis = 0.0f, sum = 0.0f;

    if (g_radius == radius && g_sigma == sigma)
        return g_table;

    XCAM_ASSERT (scale < 512);

    for (i = 0; i < scale; i++)  {
        dis = ((float)i - radius) * ((float)i - radius);
        g_table[i] = exp(-dis / (2.0f * sigma * sigma));
        sum += g_table[i];
    }

    for(i = 0; i < scale; i++)
        g_table[i] = g_table[i] / sum;

    g_radius = radius;
    g_sigma = sigma;

    return g_table;
}

static bool
gauss_blur_buffer (SmartPtr<CLBuffer> &buf, int buf_len, int g_radius, float g_sigma)
{
    float *buf_ptr = NULL;
    float *coeff = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    float *tmp_ptr = NULL;

    coeff = get_gauss_coeffs (g_radius, g_sigma);
    XCAM_ASSERT (coeff);

    ret = buf->enqueue_map((void*&)buf_ptr, 0, buf_len * sizeof (float));
    XCAM_FAIL_RETURN (ERROR, ret == XCAM_RETURN_NO_ERROR, false, "gauss_blur_buffer failed on enqueue_map");

    tmp_ptr = (float *)xcam_malloc (buf_len * sizeof (float));
    XCAM_ASSERT (tmp_ptr);
    for (int i = 0; i < buf_len; ++i) {
        tmp_ptr[i] = 0.0f;
        for (int j = -g_radius; j <= (int)g_radius; ++j) {
            tmp_ptr[i] += buf_ptr[clamp(i + j, 0, buf_len)] * coeff[g_radius + j];
        }
    }

    for (int i = 0; i < buf_len; ++i) {
        buf_ptr[i] = tmp_ptr[i];
    }
    xcam_free (tmp_ptr);
    buf->enqueue_unmap((void*)buf_ptr);
    return true;
}

PyramidLayer::PyramidLayer ()
    : blend_width (0)
    , blend_height (0)
{
    for (int plane = 0; plane < CLBlenderPlaneMax; ++plane) {
        for (int i = 0; i < XCAM_BLENDER_IMAGE_NUM; ++i) {
            gauss_offset_x[plane][i] = 0;
            lap_offset_x[plane][i] = 0;
        }
        mask_width [plane] = 0;
    }
}

CLPyramidBlender::CLPyramidBlender (
    const SmartPtr<CLContext> &context, const char *name,
    int layers, bool need_uv, bool need_seam, CLBlenderScaleMode scale_mode)
    : CLBlender (context, name, need_uv, scale_mode)
    , _layers (0)
    , _need_seam (need_seam)
    , _seam_pos_stride (0)
    , _seam_width (0)
    , _seam_height (0)
    , _seam_pos_offset_x (0)
    , _seam_pos_valid_width (0)
    , _seam_mask_done (false)
{
    if (layers <= 1)
        _layers = 1;
    else if (layers > XCAM_CL_PYRAMID_MAX_LEVEL)
        _layers = XCAM_CL_PYRAMID_MAX_LEVEL;
    else
        _layers = (uint32_t)layers;
}

CLPyramidBlender::~CLPyramidBlender ()
{
}

SmartPtr<CLImage>
CLPyramidBlender::get_gauss_image (uint32_t layer, uint32_t buf_index, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    XCAM_ASSERT (buf_index < XCAM_BLENDER_IMAGE_NUM);
    uint32_t plane = (is_uv ? 1 : 0);
    return _pyramid_layers[layer].gauss_image[plane][buf_index];
}

SmartPtr<CLImage>
CLPyramidBlender::get_lap_image (uint32_t layer, uint32_t buf_index, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    XCAM_ASSERT (buf_index < XCAM_BLENDER_IMAGE_NUM);
    uint32_t plane = (is_uv ? 1 : 0);

    return _pyramid_layers[layer].lap_image[plane][buf_index];
}

SmartPtr<CLImage>
CLPyramidBlender::get_blend_image (uint32_t layer, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    uint32_t plane = (is_uv ? 1 : 0);

    return _pyramid_layers[layer].blend_image[plane][BlendImageIndex];
}

SmartPtr<CLImage>
CLPyramidBlender::get_reconstruct_image (uint32_t layer, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    uint32_t plane = (is_uv ? 1 : 0);
    return _pyramid_layers[layer].blend_image[plane][ReconstructImageIndex];
}

SmartPtr<CLImage>
CLPyramidBlender::get_scale_image (bool is_uv)
{
    uint32_t plane = (is_uv ? 1 : 0);
    return _pyramid_layers[0].scale_image[plane];
}

SmartPtr<CLBuffer>
CLPyramidBlender::get_blend_mask (uint32_t layer, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    uint32_t plane = (is_uv ? 1 : 0);
    return _pyramid_layers[layer].blend_mask[plane];
}

SmartPtr<CLImage>
CLPyramidBlender::get_seam_mask (uint32_t layer)
{
    XCAM_ASSERT (layer < _layers);
    return _pyramid_layers[layer].seam_mask[CLSeamMaskCoeff];
}

const PyramidLayer &
CLPyramidBlender::get_pyramid_layer (uint32_t layer) const
{
    return _pyramid_layers[layer];
}

const SmartPtr<CLImage> &
CLPyramidBlender::get_image_diff () const
{
    return _image_diff;
}

void
CLPyramidBlender::get_seam_info (uint32_t &width, uint32_t &height, uint32_t &stride) const
{
    width = _seam_width;
    height = _seam_height;
    stride = _seam_pos_stride;
}

void
CLPyramidBlender::get_seam_pos_info (uint32_t &offset_x, uint32_t &valid_width) const
{
    offset_x = _seam_pos_offset_x;
    valid_width = _seam_pos_valid_width;
}

void
PyramidLayer::bind_buf_to_layer0 (
    SmartPtr<CLContext> context,
    SmartPtr<VideoBuffer> &input0, SmartPtr<VideoBuffer> &input1, SmartPtr<VideoBuffer> &output,
    const Rect &merge0_rect, const Rect &merge1_rect, bool need_uv, CLBlenderScaleMode scale_mode)
{
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    int max_plane = (need_uv ? 2 : 1);
    uint32_t divider_vert[2] = {1, 2};

    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (merge0_rect.width == merge1_rect.width);

    this->blend_width = XCAM_ALIGN_UP (merge0_rect.width, XCAM_CL_BLENDER_ALIGNMENT_X);
    this->blend_height = merge0_rect.height;

    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;

    for (int i_plane = 0; i_plane < max_plane; ++i_plane) {
        cl_desc.width = in0_info.width / 8;
        cl_desc.height = in0_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = in0_info.strides[i_plane];
        this->gauss_image[i_plane][0] = convert_to_climage (context, input0, cl_desc, in0_info.offsets[i_plane]);
        this->gauss_offset_x[i_plane][0] = merge0_rect.pos_x; // input0 offset

        cl_desc.width = in1_info.width / 8;
        cl_desc.height = in1_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = in1_info.strides[i_plane];
        this->gauss_image[i_plane][1] = convert_to_climage (context, input1, cl_desc, in1_info.offsets[i_plane]);
        this->gauss_offset_x[i_plane][1] = merge1_rect.pos_x; // input1 offset

        cl_desc.width = out_info.width / 8;
        cl_desc.height = out_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = out_info.strides[i_plane];

        if (scale_mode == CLBlenderScaleLocal) {
            this->scale_image[i_plane] = convert_to_climage (context, output, cl_desc, out_info.offsets[i_plane]);

            cl_desc.width = XCAM_ALIGN_UP (this->blend_width, XCAM_CL_BLENDER_ALIGNMENT_X) / 8;
            cl_desc.height = XCAM_ALIGN_UP (this->blend_height, divider_vert[i_plane]) / divider_vert[i_plane];
            uint32_t row_pitch = CLImage::calculate_pixel_bytes (cl_desc.format) *
                                 XCAM_ALIGN_UP (cl_desc.width, XCAM_CL_IMAGE_ALIGNMENT_X);
            uint32_t size = row_pitch * cl_desc.height;
            SmartPtr<CLBuffer> cl_buf = new CLBuffer (context, size);
            XCAM_ASSERT (cl_buf.ptr () && cl_buf->is_valid ());
            cl_desc.row_pitch = row_pitch;
            this->blend_image[i_plane][ReconstructImageIndex] = new CLImage2D (context, cl_desc, 0, cl_buf);
        } else {
            this->blend_image[i_plane][ReconstructImageIndex] =
                convert_to_climage (context, output, cl_desc, out_info.offsets[i_plane]);
        }
        XCAM_ASSERT (this->blend_image[i_plane][ReconstructImageIndex].ptr ());
    }

}

void
PyramidLayer::init_layer0 (SmartPtr<CLContext> context, bool last_layer, bool need_uv, int mask_radius, float mask_sigma)
{
    XCAM_ASSERT (this->blend_width && this->blend_height);

    //init mask
    this->mask_width[0] = this->blend_width;
    uint32_t mask_size = this->mask_width[0] * sizeof (float);
    this->blend_mask[0] = new CLBuffer(context, mask_size);
    float *blend_ptr = NULL;
    XCamReturn ret = this->blend_mask[0]->enqueue_map((void*&)blend_ptr, 0, mask_size);
    if (!xcam_ret_is_ok (ret)) {
        XCAM_LOG_ERROR ("PyramidLayer init layer0 failed in blend_mask mem_map");
        return;
    }

    for (uint32_t i_ptr = 0; i_ptr < this->mask_width[0]; ++i_ptr) {
        if (i_ptr <= this->mask_width[0] / 2)
            blend_ptr[i_ptr] = 1.0f;
        else
            blend_ptr[i_ptr] = 0.0f;
    }
    this->blend_mask[0]->enqueue_unmap ((void*)blend_ptr);
    gauss_blur_buffer (this->blend_mask[0], this->mask_width[0], mask_radius, mask_sigma);

    if (need_uv)
        copy_mask_from_y_to_uv (context);

    if (last_layer)
        return;

    int max_plane = (need_uv ? 2 : 1);
    uint32_t divider_vert[2] = {1, 2};
    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;
    for (int i_plane = 0; i_plane < max_plane; ++i_plane) {
        cl_desc.width = this->blend_width / 8;
        cl_desc.height = XCAM_ALIGN_UP (this->blend_height, divider_vert[i_plane]) / divider_vert[i_plane];

        this->blend_image[i_plane][BlendImageIndex] = new CLImage2D (context, cl_desc);
        this->lap_image[i_plane][0] = new CLImage2D (context, cl_desc);
        this->lap_image[i_plane][1] = new CLImage2D (context, cl_desc);
        this->lap_offset_x[i_plane][0] = this->lap_offset_x[i_plane][1] = 0;

#if CL_PYRAMID_ENABLE_DUMP
        this->dump_gauss_resize[i_plane] = new CLImage2D (context, cl_desc);
        this->dump_original[i_plane][0] = new CLImage2D (context, cl_desc);
        this->dump_original[i_plane][1] = new CLImage2D (context, cl_desc);
        this->dump_final[i_plane] = new CLImage2D (context, cl_desc);
#endif
    }
}

void
PyramidLayer::build_cl_images (SmartPtr<CLContext> context, bool last_layer, bool need_uv)
{
    uint32_t size = 0, row_pitch = 0;
    CLImageDesc cl_desc_set;
    SmartPtr<CLBuffer> cl_buf;
    uint32_t divider_vert[2] = {1, 2};
    uint32_t max_plane = (need_uv ? 2 : 1);

    cl_desc_set.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc_set.format.image_channel_order = CL_RGBA;

    for (uint32_t plane = 0; plane < max_plane; ++plane) {
        for (int i_image = 0; i_image < XCAM_BLENDER_IMAGE_NUM; ++i_image) {
            cl_desc_set.row_pitch = 0;
            cl_desc_set.width = XCAM_ALIGN_UP (this->blend_width, XCAM_CL_BLENDER_ALIGNMENT_X) / 8;
            cl_desc_set.height = XCAM_ALIGN_UP (this->blend_height, divider_vert[plane]) / divider_vert[plane];

            //gauss y image created by cl buffer
            row_pitch = CLImage::calculate_pixel_bytes (cl_desc_set.format) *
                        XCAM_ALIGN_UP (cl_desc_set.width, XCAM_CL_IMAGE_ALIGNMENT_X);
            size = row_pitch * cl_desc_set.height;
            cl_buf = new CLBuffer (context, size);
            XCAM_ASSERT (cl_buf.ptr () && cl_buf->is_valid ());
            cl_desc_set.row_pitch = row_pitch;
            this->gauss_image[plane][i_image] = new CLImage2D (context, cl_desc_set, 0, cl_buf);
            XCAM_ASSERT (this->gauss_image[plane][i_image].ptr ());
            this->gauss_offset_x[plane][i_image]  = 0; // offset to 0, need recalculate if for deep multi-band blender
        }

        cl_desc_set.width = XCAM_ALIGN_UP (this->blend_width, XCAM_CL_BLENDER_ALIGNMENT_X) / 8;
        cl_desc_set.height = XCAM_ALIGN_UP (this->blend_height, divider_vert[plane]) / divider_vert[plane];
        row_pitch = CLImage::calculate_pixel_bytes (cl_desc_set.format) *
                    XCAM_ALIGN_UP (cl_desc_set.width, XCAM_CL_IMAGE_ALIGNMENT_X);
        size = row_pitch * cl_desc_set.height;
        cl_buf = new CLBuffer (context, size);
        XCAM_ASSERT (cl_buf.ptr () && cl_buf->is_valid ());
        cl_desc_set.row_pitch = row_pitch;
        this->blend_image[plane][ReconstructImageIndex] = new CLImage2D (context, cl_desc_set, 0, cl_buf);
        XCAM_ASSERT (this->blend_image[plane][ReconstructImageIndex].ptr ());
#if CL_PYRAMID_ENABLE_DUMP
        this->dump_gauss_resize[plane] = new CLImage2D (context, cl_desc_set);
        this->dump_original[plane][0] = new CLImage2D (context, cl_desc_set);
        this->dump_original[plane][1] = new CLImage2D (context, cl_desc_set);
        this->dump_final[plane] = new CLImage2D (context, cl_desc_set);
#endif
        if (!last_layer) {
            cl_desc_set.row_pitch = 0;
            this->blend_image[plane][BlendImageIndex] = new CLImage2D (context, cl_desc_set);
            XCAM_ASSERT (this->blend_image[plane][BlendImageIndex].ptr ());
            for (int i_image = 0; i_image < XCAM_BLENDER_IMAGE_NUM; ++i_image) {
                this->lap_image[plane][i_image] = new CLImage2D (context, cl_desc_set);
                XCAM_ASSERT (this->lap_image[plane][i_image].ptr ());
                this->lap_offset_x[plane][i_image]  = 0; // offset to 0, need calculate from next layer if for deep multi-band blender
            }
        }
    }
}

bool
PyramidLayer::copy_mask_from_y_to_uv (SmartPtr<CLContext> &context)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (this->mask_width[0]);
    XCAM_ASSERT (this->blend_mask[0].ptr ());

    this->mask_width[1] = (this->mask_width[0] + 1) / 2;
    this->blend_mask[1] = new CLBuffer (context, this->mask_width[1] * sizeof(float));
    XCAM_ASSERT (this->blend_mask[1].ptr ());

    float *from_ptr = NULL;
    float *to_ptr = NULL;
    ret = this->blend_mask[1]->enqueue_map ((void*&)to_ptr, 0, this->mask_width[1] * sizeof(float));
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), false, "PyramidLayer copy mask failed in blend_mask[1] mem_map");
    ret = this->blend_mask[0]->enqueue_map((void*&)from_ptr, 0, this->mask_width[0] * sizeof(float));
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), false, "PyramidLayer copy mask failed in blend_mask[0] mem_map");

    for (int i = 0; i < (int)this->mask_width[1]; ++i) {
        if (i * 2 + 1 >= (int)this->mask_width[0]) { // todo i* 2 + 1
            XCAM_ASSERT (i * 2 < (int)this->mask_width[0]);
            to_ptr[i] = from_ptr[i * 2] / 2.0f;
        } else {
            to_ptr[i] = (from_ptr[i * 2] + from_ptr[i * 2 + 1]) / 2.0f;
        }
    }
    this->blend_mask[1]->enqueue_unmap ((void*)to_ptr);
    this->blend_mask[0]->enqueue_unmap ((void*)from_ptr);

    return true;
}

void
CLPyramidBlender::last_layer_buffer_redirect ()
{
    PyramidLayer &layer = _pyramid_layers[_layers - 1];
    uint32_t max_plane = (need_uv () ? 2 : 1);

    for (uint32_t plane = 0; plane < max_plane; ++plane) {
        layer.blend_image[plane][BlendImageIndex] = layer.blend_image[plane][ReconstructImageIndex];

        for (uint32_t i_image = 0; i_image < XCAM_BLENDER_IMAGE_NUM; ++i_image) {
            layer.lap_image[plane][i_image] = layer.gauss_image[plane][i_image];
        }
    }
}

void
CLPyramidBlender::dump_layer_mask (uint32_t layer, bool is_uv)
{
    const PyramidLayer &pyr_layer = get_pyramid_layer (layer);
    int plane = (is_uv ? 1 : 0);

    float *mask_ptr = NULL;
    XCamReturn ret = pyr_layer.blend_mask[plane]->enqueue_map ((void*&)mask_ptr, 0, pyr_layer.mask_width[plane] * sizeof(float));
    if (!xcam_ret_is_ok (ret)) {
        XCAM_LOG_ERROR ("CLPyramidBlender dump mask failed in blend_mask(layer:%d) mem_map", layer);
        return;
    }

    printf ("layer(%d)(-%s) mask, width:%d\n", layer, (is_uv ? "UV" : "Y"), pyr_layer.mask_width[plane]);
    for (uint32_t i = 0; i < pyr_layer.mask_width[plane]; ++i) {
        printf ("%.03f\t", mask_ptr[i]);
    }
    printf ("\n");

    pyr_layer.blend_mask[plane]->enqueue_unmap ((void*)mask_ptr);
}

static bool
gauss_fill_mask (
    SmartPtr<CLContext> context, PyramidLayer &prev, PyramidLayer &to, bool need_uv,
    int mask_radius, float mask_sigma)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    uint32_t mask_size = to.blend_width * sizeof (float);
    uint32_t prev_size = prev.mask_width[0] * sizeof (float);
    float *pre_ptr = NULL;
    int i;

    //gauss to[0]
    to.mask_width[0] = to.blend_width;
    to.blend_mask[0] = new CLBuffer (context, mask_size);
    XCAM_ASSERT (to.blend_mask[0].ptr ());
    float *mask0_ptr = NULL;
    ret = to.blend_mask[0]->enqueue_map((void*&)mask0_ptr, 0, mask_size);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), false, "gauss_fill_mask failed in destination image mem_map");

    ret = prev.blend_mask[0]->enqueue_map((void*&)pre_ptr, 0, prev_size);
    XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), false, "gauss_fill_mask failed in source image mem_map");

    for (i = 0; i < (int)to.blend_width; ++i) {
        if (i * 2 + 1 >= (int)prev.mask_width[0]) { // todo i* 2 + 1
            XCAM_ASSERT (i * 2 < (int)prev.mask_width[0]);
            mask0_ptr[i] = pre_ptr[i * 2] / 2.0f;
        } else {
            mask0_ptr[i] = (pre_ptr[i * 2] + pre_ptr[i * 2 + 1]) / 2.0f;
        }
    }
    prev.blend_mask[0]->enqueue_unmap ((void*)pre_ptr);
    to.blend_mask[0]->enqueue_unmap ((void*)mask0_ptr);

    gauss_blur_buffer (to.blend_mask[0], to.mask_width[0], mask_radius, mask_sigma);

    if (need_uv)
        to.copy_mask_from_y_to_uv (context);

    return true;
}

XCamReturn
CLPyramidBlender::allocate_cl_buffers (
    SmartPtr<CLContext> context,
    SmartPtr<VideoBuffer> &input0, SmartPtr<VideoBuffer> &input1,
    SmartPtr<VideoBuffer> &output)
{
    uint32_t index = 0;
    const Rect & window = get_merge_window ();
    bool need_reallocate = true;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    BLENDER_PROFILING_START (allocate_cl_buffers);

    need_reallocate =
        (window.width != (int32_t)_pyramid_layers[0].blend_width ||
         (window.height != 0 && window.height != (int32_t)_pyramid_layers[0].blend_height));
    _pyramid_layers[0].bind_buf_to_layer0 (
        context, input0, input1, output,
        get_input_merge_area (0), get_input_merge_area (1),
        need_uv (), get_scale_mode ());

    if (need_reallocate) {
        int g_radius = (((float)(window.width - 1) / 2) / (1 << _layers)) * 1.2f;
        float g_sigma = (float)g_radius;

        _pyramid_layers[0].init_layer0 (context, (0 == _layers - 1), need_uv(), g_radius, g_sigma);

        for (index = 1; index < _layers; ++index) {
            _pyramid_layers[index].blend_width = (_pyramid_layers[index - 1].blend_width + 1) / 2;
            _pyramid_layers[index].blend_height = (_pyramid_layers[index - 1].blend_height + 1) / 2;

            _pyramid_layers[index].build_cl_images (context, (index == _layers - 1), need_uv ());
            if (!_need_seam) {
                gauss_fill_mask (context, _pyramid_layers[index - 1], _pyramid_layers[index], need_uv (), g_radius, g_sigma);
            }
        }

        if (_need_seam) {
            ret = init_seam_buffers (context);
            XCAM_FAIL_RETURN (ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "CLPyramidBlender init seam buffer failed");
        }
    }

    //last layer buffer redirect
    last_layer_buffer_redirect ();
    _seam_mask_done = false;

    BLENDER_PROFILING_END (allocate_cl_buffers, 50);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidBlender::init_seam_buffers (SmartPtr<CLContext> context)
{
    const PyramidLayer &layer0 = get_pyramid_layer (0);
    CLImageDesc cl_desc;

    _seam_width = layer0.blend_width;
    _seam_height = layer0.blend_height;
    _seam_pos_stride = XCAM_ALIGN_UP (_seam_width, 64); // need a buffer large enough to avoid judgement in kernel
    _seam_pos_offset_x = XCAM_ALIGN_UP (_seam_width / 4, XCAM_CL_BLENDER_ALIGNMENT_X);
    if (_seam_pos_offset_x >= _seam_width)
        _seam_pos_offset_x = 0;
    _seam_pos_valid_width = XCAM_ALIGN_DOWN (_seam_width / 2, XCAM_CL_BLENDER_ALIGNMENT_X);
    if (_seam_pos_valid_width <= 0)
        _seam_pos_valid_width = XCAM_CL_BLENDER_ALIGNMENT_X;
    XCAM_ASSERT (_seam_pos_offset_x + _seam_pos_valid_width <= _seam_width);

    XCAM_ASSERT (layer0.blend_width > 0 && layer0.blend_height > 0);
    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;
    cl_desc.width = _seam_width / 8;
    cl_desc.height = _seam_height;
    cl_desc.row_pitch = CLImage::calculate_pixel_bytes (cl_desc.format) *
                        XCAM_ALIGN_UP (cl_desc.width, XCAM_CL_IMAGE_ALIGNMENT_X);

    uint32_t image_diff_size = cl_desc.row_pitch * _seam_height;
    SmartPtr<CLBuffer> cl_diff_buf = new CLBuffer (context, image_diff_size);
    XCAM_FAIL_RETURN (
        ERROR,
        cl_diff_buf.ptr () && cl_diff_buf->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidBlender init seam buffer failed to create image_difference buffers");

    _image_diff = new CLImage2D (context, cl_desc, 0, cl_diff_buf);
    XCAM_FAIL_RETURN (
        ERROR,
        _image_diff.ptr () && _image_diff->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidBlender init seam buffer failed to bind image_difference data");

    uint32_t pos_buf_size = sizeof (SEAM_POS_TYPE) * _seam_pos_stride * _seam_height;
    uint32_t sum_buf_size = sizeof (SEAM_SUM_TYPE) * _seam_pos_stride * 2; // 2 lines
    _seam_pos_buf = new CLBuffer (context, pos_buf_size, CL_MEM_READ_WRITE);
    _seam_sum_buf = new CLBuffer (context, sum_buf_size, CL_MEM_READ_WRITE);
    XCAM_FAIL_RETURN (
        ERROR,
        _seam_pos_buf.ptr () && _seam_pos_buf->is_valid () &&
        _seam_sum_buf.ptr () && _seam_sum_buf->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidBlender init seam buffer failed to create seam buffers");

    uint32_t mask_width = XCAM_ALIGN_UP(_seam_width, XCAM_CL_BLENDER_ALIGNMENT_X);
    uint32_t mask_height = XCAM_ALIGN_UP(_seam_height, 2);
    for (uint32_t i = 0; i < _layers; ++i) {
        cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
        cl_desc.format.image_channel_order = CL_RGBA;
        cl_desc.width = mask_width / 8;
        cl_desc.height = mask_height;
        cl_desc.row_pitch = CLImage::calculate_pixel_bytes (cl_desc.format) *
                            XCAM_ALIGN_UP (cl_desc.width, XCAM_CL_IMAGE_ALIGNMENT_X);

        uint32_t mask_size = cl_desc.row_pitch * mask_height;
        SmartPtr<CLBuffer> cl_buf0 = new CLBuffer (context, mask_size);
        SmartPtr<CLBuffer> cl_buf1 = new CLBuffer (context, mask_size);
        XCAM_ASSERT (cl_buf0.ptr () && cl_buf0->is_valid () && cl_buf1.ptr () && cl_buf1->is_valid ());

        _pyramid_layers[i].seam_mask[CLSeamMaskTmp] = new CLImage2D (context, cl_desc, 0, cl_buf0);
        _pyramid_layers[i].seam_mask[CLSeamMaskCoeff] = new CLImage2D (context, cl_desc, 0, cl_buf1);
        XCAM_FAIL_RETURN (
            ERROR,
            _pyramid_layers[i].seam_mask[CLSeamMaskTmp].ptr () && _pyramid_layers[i].seam_mask[CLSeamMaskTmp]->is_valid () &&
            _pyramid_layers[i].seam_mask[CLSeamMaskCoeff].ptr () && _pyramid_layers[i].seam_mask[CLSeamMaskCoeff]->is_valid (),
            XCAM_RETURN_ERROR_CL,
            "CLPyramidBlender init seam buffer failed to create seam_mask buffer");

        mask_width = XCAM_ALIGN_UP(mask_width / 2, XCAM_CL_BLENDER_ALIGNMENT_X);
        mask_height = XCAM_ALIGN_UP(mask_height / 2, 2);
    }

    return XCAM_RETURN_NO_ERROR;
}

static void
assign_mask_line (SEAM_MASK_TYPE *mask_ptr, int line, int stride, int delimiter)
{
#define MASK_1 0xFFFF
#define MASK_0 0x00

    SEAM_MASK_TYPE *line_ptr = mask_ptr + line * stride;
    int mask_1_len = delimiter + 1;

    memset (line_ptr, MASK_1, sizeof (SEAM_MASK_TYPE) * mask_1_len);
    memset (line_ptr + mask_1_len, MASK_0, sizeof (SEAM_MASK_TYPE) * (stride - mask_1_len));
}

XCamReturn
CLPyramidBlender::fill_seam_mask ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    XCAM_ASSERT (_seam_pos_buf.ptr () && _seam_sum_buf.ptr ());
    uint32_t pos_buf_size = sizeof (SEAM_POS_TYPE) * _seam_pos_stride * _seam_height;
    uint32_t sum_buf_size = sizeof (SEAM_SUM_TYPE) * _seam_pos_stride * 2;
    SEAM_SUM_TYPE *sum_ptr;
    SEAM_POS_TYPE *pos_ptr;
    SEAM_MASK_TYPE *mask_ptr;

    if (_seam_mask_done)
        return XCAM_RETURN_NO_ERROR;

    ret = _seam_sum_buf->enqueue_map ((void *&)sum_ptr, 0, sum_buf_size, CL_MAP_READ);
    XCAM_FAIL_RETURN (ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "CLPyramidBlender map seam_sum_buf failed");

    float min_sum = 9999999999.0f, tmp_sum;
    int pos = 0, min_pos0, min_pos1;
    int i = 0;
    SEAM_SUM_TYPE *sum_ptr0 = sum_ptr, *sum_ptr1 = sum_ptr + _seam_pos_stride;
    for (i = (int)_seam_pos_offset_x; i < (int)(_seam_pos_offset_x + _seam_pos_valid_width); ++i) {
        tmp_sum = sum_ptr0[i] + sum_ptr1[i];
        if (tmp_sum >= min_sum)
            continue;
        pos = (int)i;
        min_sum = tmp_sum;
    }
    _seam_sum_buf->enqueue_unmap ((void*)sum_ptr);
    min_pos0 = min_pos1 = pos;

    BLENDER_PROFILING_START (fill_seam_mask);

    // reset layer0 seam_mask
    SmartPtr<CLImage> seam_mask = _pyramid_layers[0].seam_mask[CLSeamMaskTmp];
    const CLImageDesc &mask_desc = seam_mask->get_image_desc ();
    size_t mask_origin[3] = {0, 0, 0};
    size_t mask_region[3] = {mask_desc.width, mask_desc.height, 1};
    size_t mask_row_pitch;
    size_t mask_slice_pitch;
    ret = seam_mask->enqueue_map ((void *&)mask_ptr, mask_origin, mask_region,
                                  &mask_row_pitch, &mask_slice_pitch, CL_MAP_READ);
    XCAM_FAIL_RETURN (ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "CLPyramidBlender map seam_mask failed");
    uint32_t mask_stride = mask_row_pitch / sizeof (SEAM_MASK_TYPE);
    ret = _seam_pos_buf->enqueue_map ((void *&)pos_ptr, 0, pos_buf_size, CL_MAP_READ);
    XCAM_FAIL_RETURN (ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "CLPyramidBlender map seam_pos_buf failed");
    //printf ("***********min sum:%.3f, pos:%d, sum0:%.3f, sum1:%.3f\n", min_sum, pos, sum_ptr0[pos], sum_ptr1[pos]);
    for (i = _seam_height / 2 - 1; i >= 0; --i) {
        assign_mask_line (mask_ptr, i, mask_stride, min_pos0);
        min_pos0 = pos_ptr [i * _seam_pos_stride + min_pos0];
    }

    for (i = _seam_height / 2; i < (int)_seam_height; ++i) {
        assign_mask_line (mask_ptr, i, mask_stride, min_pos1);
        min_pos1 = pos_ptr [i * _seam_pos_stride + min_pos1];
    }
    for (; i < (int)mask_desc.height; ++i) {
        assign_mask_line (mask_ptr, i, mask_stride, min_pos1);
    }

    seam_mask->enqueue_unmap ((void*)mask_ptr);
    _seam_pos_buf->enqueue_unmap ((void*)pos_ptr);

    _seam_mask_done = true;

    BLENDER_PROFILING_END (fill_seam_mask, 50);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidBlender::execute_done (SmartPtr<VideoBuffer> &output)
{
    int max_plane = (need_uv () ? 2 : 1);
    XCAM_UNUSED (output);

#if CL_PYRAMID_ENABLE_DUMP
    dump_buffers ();
#endif

    for (int i_plane = 0; i_plane < max_plane; ++i_plane) {
        _pyramid_layers[0].gauss_image[i_plane][0].release ();
        _pyramid_layers[0].gauss_image[i_plane][1].release ();
        _pyramid_layers[0].blend_image[i_plane][ReconstructImageIndex].release ();

        if (_layers <= 1) {
            _pyramid_layers[_layers - 1].blend_image[i_plane][BlendImageIndex].release ();
            _pyramid_layers[_layers - 1].lap_image[i_plane][0].release ();
            _pyramid_layers[_layers - 1].lap_image[i_plane][1].release ();
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

CLPyramidBlendKernel::CLPyramidBlendKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, bool is_uv, bool need_seam)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _is_uv (is_uv)
    , _need_seam (need_seam)
{
}

XCamReturn
CLPyramidBlendKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in0 = get_input_0 ();
    SmartPtr<CLImage> image_in1 = get_input_1 ();
    SmartPtr<CLImage> image_out = get_output ();
    SmartPtr<CLMemory> buf_mask;
    if (_need_seam)
        buf_mask = get_seam_mask ();
    else
        buf_mask = get_blend_mask ();

    XCAM_ASSERT (image_in0.ptr () && image_in1.ptr () && image_out.ptr ());
    const CLImageDesc &cl_desc_out = image_out->get_image_desc ();

    args.push_back (new CLMemArgument (image_in0));
    args.push_back (new CLMemArgument (image_in1));
    args.push_back (new CLMemArgument (buf_mask));
    args.push_back (new CLMemArgument (image_out));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 8;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out.height, work_size.local[1]);
    return XCAM_RETURN_NO_ERROR;
}

CLPyramidTransformKernel::CLPyramidTransformKernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    uint32_t buf_index,
    bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _buf_index (buf_index)
    , _is_uv (is_uv)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
    XCAM_ASSERT (buf_index <= XCAM_BLENDER_IMAGE_NUM);
}

static bool
change_image_format (
    SmartPtr<CLContext> context, SmartPtr<CLImage> input,
    SmartPtr<CLImage> &output, const CLImageDesc &new_desc)
{
    SmartPtr<CLImage2D> previous = input.dynamic_cast_ptr<CLImage2D> ();
    if (!previous.ptr () || !previous->get_bind_buf ().ptr ())
        return false;

    SmartPtr<CLBuffer> bind_buf = previous->get_bind_buf ();
    output = new CLImage2D (context, new_desc, 0, bind_buf);
    if (!output.ptr ())
        return false;
    return true;
}

int32_t
CLPyramidTransformKernel::get_input_gauss_offset_x ()
{
    const PyramidLayer &layer = _blender->get_pyramid_layer (_layer);
    uint32_t plane_index = (_is_uv ? 1 : 0);
    return layer.gauss_offset_x[plane_index][_buf_index];
}

XCamReturn
CLPyramidTransformKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in_gauss = get_input_gauss();
    SmartPtr<CLImage> image_out_gauss = get_output_gauss();
    //SmartPtr<CLImage> image_out_lap = get_output_lap ();
    const CLImageDesc &cl_desc_out_gauss_pre = image_out_gauss->get_image_desc ();

    CLImageDesc cl_desc_out_gauss;
    cl_desc_out_gauss.format.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_desc_out_gauss.format.image_channel_order = CL_RGBA;
    cl_desc_out_gauss.width = cl_desc_out_gauss_pre.width * 2;
    cl_desc_out_gauss.height = cl_desc_out_gauss_pre.height;
    cl_desc_out_gauss.row_pitch = cl_desc_out_gauss_pre.row_pitch;
    SmartPtr<CLImage> format_image_out;
    change_image_format (context, image_out_gauss, format_image_out, cl_desc_out_gauss);
    XCAM_FAIL_RETURN (
        ERROR,
        format_image_out.ptr () && format_image_out->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change output gauss image format failed");

    int gauss_offset_x = get_input_gauss_offset_x () / 8;
    XCAM_ASSERT (gauss_offset_x * 8 == get_input_gauss_offset_x ());

    args.push_back (new CLMemArgument (image_in_gauss));
    args.push_back (new CLArgumentT<int> (gauss_offset_x));
    args.push_back (new CLMemArgument (format_image_out));

#if CL_PYRAMID_ENABLE_DUMP
    int plane = _is_uv ? 1 : 0;
    SmartPtr<CLImage> dump_original = _blender->get_pyramid_layer (_layer).dump_original[plane][_buf_index];

    args.push_back (new CLMemArgument (dump_original));

    printf ("L%dI%d: gauss_offset_x:%d \n", _layer, _buf_index, gauss_offset_x);
#endif

    const int workitem_lines = 2;
    int gloabal_y = XCAM_ALIGN_UP (cl_desc_out_gauss.height, workitem_lines) / workitem_lines;
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out_gauss.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (gloabal_y, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLSeamDiffKernel::CLSeamDiffKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender)
    : CLImageKernel (context)
    , _blender (blender)
{
}

XCamReturn
CLSeamDiffKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    const PyramidLayer &layer0 = _blender->get_pyramid_layer (0);
    SmartPtr<CLImage> image0 = layer0.gauss_image[CLBlenderPlaneY][0];
    SmartPtr<CLImage> image1 = layer0.gauss_image[CLBlenderPlaneY][1];
    SmartPtr<CLImage> out_diff = _blender->get_image_diff ();
    CLImageDesc out_diff_desc = out_diff->get_image_desc ();

    int image_offset_x[XCAM_BLENDER_IMAGE_NUM];

    for (uint32_t i = 0; i < XCAM_BLENDER_IMAGE_NUM; ++i) {
        image_offset_x[i] = layer0.gauss_offset_x[CLBlenderPlaneY][i] / 8;
    }

    args.push_back (new CLMemArgument (image0));
    args.push_back (new CLArgumentT<int> (image_offset_x[0]));
    args.push_back (new CLMemArgument (image1));
    args.push_back (new CLArgumentT<int> (image_offset_x[1]));
    args.push_back (new CLMemArgument (out_diff));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (out_diff_desc.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (out_diff_desc.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLSeamDPKernel::CLSeamDPKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender)
    : CLImageKernel (context)
    , _blender (blender)
{
}

XCamReturn
CLSeamDPKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
#define ELEMENT_PIXEL 1

    uint32_t width, height, stride;
    uint32_t pos_offset_x, pos_valid_width;
    _blender->get_seam_info (width, height, stride);
    _blender->get_seam_pos_info (pos_offset_x, pos_valid_width);
    int seam_height = (int)height;
    int seam_stride = (int)stride / ELEMENT_PIXEL;
    int seam_offset_x = (int)pos_offset_x / ELEMENT_PIXEL; // ushort8
    int seam_valid_with = (int)pos_valid_width / ELEMENT_PIXEL;
    int max_pos = (int)(pos_offset_x + pos_valid_width - 1);

    SmartPtr<CLImage> image = _blender->get_image_diff ();
    SmartPtr<CLBuffer> pos_buf = _blender->get_seam_pos_buf ();
    SmartPtr<CLBuffer> sum_buf = _blender->get_seam_sum_buf ();
    XCAM_ASSERT (image.ptr () && pos_buf.ptr () && sum_buf.ptr ());

    CLImageDesc cl_orig = image->get_image_desc ();
    CLImageDesc cl_desc_convert;
    cl_desc_convert.format.image_channel_data_type = CL_UNSIGNED_INT8;
    cl_desc_convert.format.image_channel_order = CL_R;
    cl_desc_convert.width = cl_orig.width * (8 / ELEMENT_PIXEL);
    cl_desc_convert.height = cl_orig.height;
    cl_desc_convert.row_pitch = cl_orig.row_pitch;

    SmartPtr<CLImage> convert_image;
    change_image_format (get_context (), image, convert_image, cl_desc_convert);
    XCAM_ASSERT (convert_image.ptr () && convert_image->is_valid ());

    args.push_back (new CLMemArgument (convert_image));
    args.push_back (new CLMemArgument (pos_buf));
    args.push_back (new CLMemArgument (sum_buf));
    args.push_back (new CLArgumentT<int> (seam_offset_x));
    args.push_back (new CLArgumentT<int> (seam_valid_with));
    args.push_back (new CLArgumentT<int> (max_pos));
    args.push_back (new CLArgumentT<int> (seam_height));
    args.push_back (new CLArgumentT<int> (seam_stride));

    work_size.dim = 1;
    work_size.local[0] = XCAM_ALIGN_UP(seam_valid_with, 16);
    work_size.global[0] = work_size.local[0] * 2;

    return XCAM_RETURN_NO_ERROR;
}

CLPyramidSeamMaskKernel::CLPyramidSeamMaskKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, bool scale, bool need_slm)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _need_scale (scale)
    , _need_slm (need_slm)
{
    XCAM_ASSERT (layer < blender->get_layers ());
}

XCamReturn
CLPyramidSeamMaskKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    ret = _blender->fill_seam_mask ();
    XCAM_FAIL_RETURN (ERROR, ret == XCAM_RETURN_NO_ERROR, ret, "CLPyramidSeamMaskKernel fill seam mask failed");

    SmartPtr<CLContext> context = get_context ();
    const PyramidLayer &cur_layer = _blender->get_pyramid_layer (_layer);
    SmartPtr<CLImage> input_image = cur_layer.seam_mask[CLSeamMaskTmp];
    SmartPtr<CLImage> out_gauss = cur_layer.seam_mask[CLSeamMaskCoeff];
    CLImageDesc out_gauss_desc = out_gauss->get_image_desc ();

    XCAM_ASSERT (input_image.ptr () && out_gauss.ptr ());
    XCAM_ASSERT (input_image->is_valid () && out_gauss->is_valid ());

    args.push_back (new CLMemArgument (input_image));
    args.push_back (new CLMemArgument (out_gauss));



    if (_need_slm) {
        int image_width = out_gauss_desc.width;
        args.push_back (new CLArgumentT<int> (image_width));
    }

    if (_need_scale) {
        const PyramidLayer &next_layer = _blender->get_pyramid_layer (_layer + 1);
        SmartPtr<CLImage> out_orig = next_layer.seam_mask[CLSeamMaskTmp];
        CLImageDesc input_desc, output_desc;
        input_desc = out_orig->get_image_desc ();
        output_desc.format.image_channel_data_type = CL_UNSIGNED_INT8;
        output_desc.format.image_channel_order = CL_RGBA;
        output_desc.width = input_desc.width * 2;
        output_desc.height = input_desc.height;
        output_desc.row_pitch = input_desc.row_pitch;

        SmartPtr<CLImage> output_scale_image;
        change_image_format (context, out_orig, output_scale_image, output_desc);
        args.push_back (new CLMemArgument (output_scale_image));
    }

    uint32_t workitem_height = XCAM_ALIGN_UP (out_gauss_desc.height, 2) / 2;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;

    if (_need_slm) {
        work_size.local[0] = XCAM_ALIGN_UP (out_gauss_desc.width, 16);
        work_size.local[1] = 1;
        work_size.global[0] = work_size.local[0];
        work_size.global[1] = workitem_height;
    } else {
        work_size.local[0] = 8;
        work_size.local[1] = 4;
        work_size.global[0] = XCAM_ALIGN_UP (out_gauss_desc.width, work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (workitem_height, work_size.local[1]);
    }

    return XCAM_RETURN_NO_ERROR;
}

CLPyramidLapKernel::CLPyramidLapKernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    uint32_t buf_index,
    bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _buf_index (buf_index)
    , _is_uv (is_uv)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
    XCAM_ASSERT (buf_index <= XCAM_BLENDER_IMAGE_NUM);
}

int32_t
CLPyramidLapKernel::get_cur_gauss_offset_x ()
{
    const PyramidLayer &layer = _blender->get_pyramid_layer (_layer);
    uint32_t plane_index = (_is_uv ? 1 : 0);
    return layer.gauss_offset_x[plane_index][_buf_index];
}

int32_t
CLPyramidLapKernel::get_output_lap_offset_x ()
{
    const PyramidLayer &layer = _blender->get_pyramid_layer (_layer);
    uint32_t plane_index = (_is_uv ? 1 : 0);
    return layer.lap_offset_x[plane_index][_buf_index];
}

XCamReturn
CLPyramidLapKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> cur_gauss_image = get_current_gauss();
    SmartPtr<CLImage> next_gauss_image_tmp = get_next_gauss();
    SmartPtr<CLImage> image_out_lap = get_output_lap ();
    const CLImageDesc &cl_desc_next_gauss_tmp = next_gauss_image_tmp->get_image_desc ();
    const CLImageDesc &cl_desc_out_lap = image_out_lap->get_image_desc ();
    float next_gauss_pixel_width = 0.0f, next_gauss_pixel_height = 0.0f;

    CLImageDesc cl_desc_next_gauss;
    if (!_is_uv) {
        cl_desc_next_gauss.format.image_channel_data_type = CL_UNORM_INT8;
        cl_desc_next_gauss.format.image_channel_order = CL_R;
        cl_desc_next_gauss.width = cl_desc_next_gauss_tmp.width * 8;
    } else {
        cl_desc_next_gauss.format.image_channel_data_type = CL_UNORM_INT8;
        cl_desc_next_gauss.format.image_channel_order = CL_RG;
        cl_desc_next_gauss.width = cl_desc_next_gauss_tmp.width * 4;
    }
    cl_desc_next_gauss.height = cl_desc_next_gauss_tmp.height;
    cl_desc_next_gauss.row_pitch = cl_desc_next_gauss_tmp.row_pitch;
    SmartPtr<CLImage> next_gauss;
    change_image_format (context, next_gauss_image_tmp, next_gauss, cl_desc_next_gauss);
    XCAM_FAIL_RETURN (
        ERROR,
        next_gauss.ptr () && next_gauss->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change output gauss image format failed");

    next_gauss_pixel_width = cl_desc_next_gauss.width;
    next_gauss_pixel_height = cl_desc_next_gauss.height;

    // out format(current layer): CL_UNSIGNED_INT16 + CL_RGBA
    float out_width = CLImage::calculate_pixel_bytes (cl_desc_next_gauss.format) * cl_desc_next_gauss.width * 2.0f / 8.0f;
    float out_height = next_gauss_pixel_height * 2.0f;
    float sampler_offset_x = SAMPLER_POSITION_OFFSET / next_gauss_pixel_width;
    float sampler_offset_y = SAMPLER_POSITION_OFFSET / next_gauss_pixel_height;

    int cur_gauss_offset_x = get_cur_gauss_offset_x () / 8;
    XCAM_ASSERT (cur_gauss_offset_x * 8 == get_cur_gauss_offset_x ());
    int lap_offset_x = get_output_lap_offset_x () / 8;
    XCAM_ASSERT (lap_offset_x * 8 == get_output_lap_offset_x ());

    args.push_back (new CLMemArgument (cur_gauss_image));
    args.push_back (new CLArgumentT<int> (cur_gauss_offset_x));
    args.push_back (new CLMemArgument (next_gauss));
    args.push_back (new CLArgumentT<float> (sampler_offset_x));
    args.push_back (new CLArgumentT<float> (sampler_offset_y));
    args.push_back (new CLMemArgument (image_out_lap));
    args.push_back (new CLArgumentT<int> (lap_offset_x));
    args.push_back (new CLArgumentT<float> (out_width));
    args.push_back (new CLArgumentT<float> (out_height));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out_lap.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out_lap.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLPyramidReconstructKernel::CLPyramidReconstructKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _is_uv (is_uv)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
}

int
CLPyramidReconstructKernel::get_output_reconstrcut_offset_x ()
{
    if (_layer > 0)
        return 0;
    const Rect & window = _blender->get_merge_window ();
    XCAM_ASSERT (window.pos_x % XCAM_CL_BLENDER_ALIGNMENT_X == 0);
    return window.pos_x;
}

XCamReturn
CLPyramidReconstructKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in_reconst = get_input_reconstruct();
    SmartPtr<CLImage> image_in_lap = get_input_lap ();
    SmartPtr<CLImage> image_out_reconst = get_output_reconstruct();
    const CLImageDesc &cl_desc_in_reconst_pre = image_in_reconst->get_image_desc ();
    // out_desc should be same as image_in_lap
    const CLImageDesc &cl_desc_out_reconst = image_in_lap->get_image_desc (); // don't change
    float input_gauss_width = 0.0f, input_gauss_height = 0.0f;

    CLImageDesc cl_desc_in_reconst;
    cl_desc_in_reconst.format.image_channel_data_type = CL_UNORM_INT8;
    if (_is_uv) {
        cl_desc_in_reconst.format.image_channel_order = CL_RG;
        cl_desc_in_reconst.width = cl_desc_in_reconst_pre.width * 4;
    } else {
        cl_desc_in_reconst.format.image_channel_order = CL_R;
        cl_desc_in_reconst.width = cl_desc_in_reconst_pre.width * 8;
    }
    cl_desc_in_reconst.height = cl_desc_in_reconst_pre.height;
    cl_desc_in_reconst.row_pitch = cl_desc_in_reconst_pre.row_pitch;
    SmartPtr<CLImage> input_reconstruct;
    change_image_format (context, image_in_reconst, input_reconstruct, cl_desc_in_reconst);
    XCAM_FAIL_RETURN (
        ERROR,
        input_reconstruct.ptr () && input_reconstruct->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change output gauss image format failed");

    input_gauss_width = cl_desc_in_reconst.width;
    input_gauss_height = cl_desc_in_reconst.height;

    float out_reconstruct_width = CLImage::calculate_pixel_bytes (cl_desc_in_reconst.format) * cl_desc_in_reconst.width * 2.0f / 8.0f;
    float out_reconstruct_height = input_gauss_height * 2.0f;
    float in_sampler_offset_x = SAMPLER_POSITION_OFFSET / input_gauss_width;
    float in_sampler_offset_y = SAMPLER_POSITION_OFFSET / input_gauss_height;
    int out_reconstruct_offset_x = 0;

    if (_blender->get_scale_mode () == CLBlenderScaleLocal) {
        out_reconstruct_offset_x = 0;
    } else {
        out_reconstruct_offset_x = get_output_reconstrcut_offset_x () / 8;
        XCAM_ASSERT (out_reconstruct_offset_x * 8 == get_output_reconstrcut_offset_x ());
    }

    args.push_back (new CLMemArgument (input_reconstruct));
    args.push_back (new CLArgumentT<float> (in_sampler_offset_x));
    args.push_back (new CLArgumentT<float> (in_sampler_offset_y));
    args.push_back (new CLMemArgument (image_in_lap));
    args.push_back (new CLMemArgument (image_out_reconst));
    args.push_back (new CLArgumentT<int> (out_reconstruct_offset_x));
    args.push_back (new CLArgumentT<float> (out_reconstruct_width));
    args.push_back (new CLArgumentT<float> (out_reconstruct_height));

#if CL_PYRAMID_ENABLE_DUMP
    int i_plane = (_is_uv ? 1 : 0);
    const PyramidLayer &cur_layer = _blender->get_pyramid_layer (_layer);
    SmartPtr<CLImage>  dump_gauss_resize = cur_layer.dump_gauss_resize[i_plane];
    SmartPtr<CLImage>  dump_final = cur_layer.dump_final[i_plane];

    args.push_back (new CLMemArgument (dump_gauss_resize));
    args.push_back (new CLMemArgument (dump_final));

    printf ("Rec%d: reconstruct_offset_x:%d, out_width:%.2f, out_height:%.2f, in_sampler_offset_x:%.2f, in_sampler_offset_y:%.2f\n",
            _layer, out_reconstruct_offset_x, out_reconstruct_width, out_reconstruct_height,
            in_sampler_offset_x, in_sampler_offset_y);
#endif

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 4;
    work_size.local[1] = 8;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out_reconst.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out_reconst.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}


void
CLPyramidBlender::dump_buffers ()
{
    static int frame_count = 0;
    SmartPtr<CLImage> image;
    ++frame_count;

    // dump difference between original image and final image
#if 0
#define CM_NUM 3
    SmartPtr<CLImage> images[CM_NUM];
    const Rect & window = get_merge_window ();
    int offsets[3] = {window.pos_x, window.pos_x, 0};
    //right edge
    //int offsets[3] = {0 + window.width - 8, window.pos_x + window.width - 8, window.width - 8};
    size_t row_pitch[CM_NUM];
    size_t slice_pitch[CM_NUM];
    uint8_t *ptr[CM_NUM] = {NULL, NULL, NULL};
    uint32_t i = 0;

#if 1
    // Y
    // left edge
    images[0] = this->get_pyramid_layer (0).gauss_image[0][0];
    // right edge
    //images[0] = this->get_pyramid_layer (0).gauss_image[0][1];
    images[1] = this->get_pyramid_layer (0).blend_image[0][ReconstructImageIndex];
    images[2] = this->get_pyramid_layer (0).dump_final[0];
#else
    // UV
    // left edge
    images[0] = this->get_pyramid_layer (0).gauss_image[1][0];
    // right edge
    //images[0] = this->get_pyramid_layer (0).gauss_image[1][1];
    images[1] = this->get_pyramid_layer (0).blend_image[1][ReconstructImageIndex];
    images[2] = this->get_pyramid_layer (0).dump_final[1];
#endif

    for (i = 0; i < CM_NUM; ++i) {
        const CLImageDesc &desc = images[i]->get_image_desc ();
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {desc.width, desc.height, 1};
        XCamReturn ret = images[i]->enqueue_map ((void *&)ptr[i], origin, region, &row_pitch[i], &slice_pitch[i], CL_MAP_READ);
        XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);
    }
    // offset UV, workaround of beignet
    //offsets[0] += row_pitch[0] * 1088;
    //offsets[1] += row_pitch[1] * 1088;

    printf ("layer 0(UV) comparison, original / final-image / reconstruct offset:%d, width:%d\n", window.pos_x, window.width);
    for (int ih = 250; ih < 280; ++ih) {
        uint8_t *lines[CM_NUM];
        for (i = 0; i < 2 /*CM_NUM*/; ++i) {
            uint8_t *l = (uint8_t *)ptr[i] + offsets[i] + row_pitch[i] * ih + 0;
            lines[i] = l;
            printf ("%02x%02x%02x%02x%02x%02x%02x%02x ", l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7]);
        }
        //printf differrence between original and final image
        printf ("delta(orig - final):");
        for (i = 0; i < 10; ++i) {
            printf ("%02x", (uint32_t)(lines[0][i] - lines[1][i]) & 0xFF);
        }
        printf ("\n");
    }

    for (i = 0; i < CM_NUM; ++i) {
        images[i]->enqueue_unmap (ptr[i]);
    }
#endif

#define DUMP_IMAGE(prefix, image, layer)        \
    desc = (image)->get_image_desc ();   \
    snprintf (filename, sizeof(filename), prefix "_L%d-%dx%d",            \
              layer, (image)->get_pixel_bytes () * desc.width, desc.height); \
    dump_image (image, filename)

    // dump image data to file
    CLImageDesc desc;
    char filename[1024];

    image = this->get_image_diff ();
    if (image.ptr ()) {
        DUMP_IMAGE ("dump_image_diff", image, 0);
    }

    for (uint32_t i_layer = 0; i_layer < get_layers (); ++i_layer) {
        //dump seam mask
        image = this->get_pyramid_layer(i_layer).seam_mask[CLSeamMaskTmp];
        if (image.ptr ()) {
            DUMP_IMAGE ("dump_seam_tmp", image, i_layer);
        }

        image = this->get_pyramid_layer(i_layer).seam_mask[CLSeamMaskCoeff];
        if (image.ptr ()) {
            DUMP_IMAGE ("dump_seam_coeff", image, i_layer);
        }

        image = this->get_blend_image (i_layer, false); // layer 1
        DUMP_IMAGE ("dump_blend", image, i_layer);

        if (i_layer > 0) { //layer : [1, _layers -1]
            image = this->get_gauss_image (i_layer, 0, false);
            DUMP_IMAGE ("dump_gaussI0", image, i_layer);
            image = this->get_gauss_image (i_layer, 1, false);
            DUMP_IMAGE ("dump_gaussI1", image, i_layer);
        }

        if (i_layer < get_layers () - 1) {
            image = this->get_lap_image (i_layer, 0, false); // layer : [0, _layers -2]
            DUMP_IMAGE ("dump_lap_I0", image, i_layer);
        }
    }

#if CL_PYRAMID_ENABLE_DUMP
    image = this->get_pyramid_layer (0).dump_gauss_resize[0];
    DUMP_IMAGE ("dump_gauss_resize", image, 0);

    image = this->get_pyramid_layer (0).dump_original[0][0];
    DUMP_IMAGE ("dump_orginalI0", image, 0);
    image = this->get_pyramid_layer (0).dump_original[0][1];
    DUMP_IMAGE ("dump_orginalI1", image, 0);

    image = this->get_pyramid_layer (0).dump_final[CLBlenderPlaneY];
    DUMP_IMAGE ("dump_final", image, 0);
#endif

#if 0
    this->dump_layer_mask (0, false);
    this->dump_layer_mask (1, false);

    //this->dump_layer_mask (0, true);
    //this->dump_layer_mask (1, true);
#endif

}

CLBlenderLocalScaleKernel::CLBlenderLocalScaleKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, bool is_uv)
    : CLBlenderScaleKernel (context, is_uv)
    , _blender (blender)
{
}

SmartPtr<CLImage>
CLBlenderLocalScaleKernel::get_input_image ()
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> rec_image = _blender->get_reconstruct_image (0, _is_uv);
    const CLImageDesc &rec_desc = rec_image->get_image_desc ();

    CLImageDesc new_desc;
    new_desc.format.image_channel_data_type = CL_UNORM_INT8;
    if (_is_uv) {
        new_desc.format.image_channel_order = CL_RG;
        new_desc.width = rec_desc.width * 4;
    } else {
        new_desc.format.image_channel_order = CL_R;
        new_desc.width = rec_desc.width * 8;
    }
    new_desc.height = rec_desc.height;
    new_desc.row_pitch = rec_desc.row_pitch;
    SmartPtr<CLImage> new_image;
    change_image_format (context, rec_image, new_image, new_desc);
    XCAM_FAIL_RETURN (
        ERROR,
        new_image.ptr () && new_image->is_valid (),
        NULL,
        "CLBlenderLocalScaleKernel change image format failed");

    _image_in = new_image;
    return new_image;
}

SmartPtr<CLImage>
CLBlenderLocalScaleKernel::get_output_image ()
{
    return _blender->get_scale_image (_is_uv);
}

bool
CLBlenderLocalScaleKernel::get_output_info (
    uint32_t &out_width, uint32_t &out_height, int &out_offset_x)
{
    XCAM_ASSERT (_image_in.ptr ());

    const Rect &window = _blender->get_merge_window ();
    const CLImageDesc &desc_in = _image_in->get_image_desc ();

    out_width = window.width / 8;
    out_height = desc_in.height;
    out_offset_x = window.pos_x / 8;

    XCAM_FAIL_RETURN (ERROR, out_width != 0, false, "get output info failed");
    return true;
}

CLPyramidCopyKernel::CLPyramidCopyKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t buf_index, bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _is_uv (is_uv)
    , _buf_index (buf_index)
{
}

XCamReturn
CLPyramidCopyKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> from = get_input ();
    SmartPtr<CLImage> to = get_output ();

    const CLImageDesc &to_desc = to->get_image_desc ();
    const Rect &window = _blender->get_merge_window ();
    const Rect &input_area = _blender->get_input_valid_area (_buf_index);
    const Rect &merge_area = _blender->get_input_merge_area (_buf_index);
    int in_offset_x = 0;
    int out_offset_x = 0;
    int max_g_x = 0, max_g_y = 0;

    if (_buf_index == 0) {
        in_offset_x = input_area.pos_x / 8;
        max_g_x = (merge_area.pos_x - input_area.pos_x) / 8;
        out_offset_x = window.pos_x / 8 - max_g_x;
    } else {
        in_offset_x = (merge_area.pos_x + merge_area.width) / 8;
        out_offset_x = (window.pos_x + window.width) / 8;
        max_g_x = (input_area.pos_x + input_area.width) / 8 - in_offset_x;
    }
    max_g_y = to_desc.height;
    XCAM_ASSERT (max_g_x > 0 && max_g_x <= (int)to_desc.width);

#if CL_PYRAMID_ENABLE_DUMP
    printf ("copy(%d), in_offset_x:%d, out_offset_x:%d, max_x:%d\n", _buf_index, in_offset_x, out_offset_x, max_g_x);
#endif

    args.push_back (new CLMemArgument (from));
    args.push_back (new CLArgumentT<int> (in_offset_x));
    args.push_back (new CLMemArgument (to));
    args.push_back (new CLArgumentT<int> (out_offset_x));
    args.push_back (new CLArgumentT<int> (max_g_x));
    args.push_back (new CLArgumentT<int> (max_g_y));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (max_g_x, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (max_g_y, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<CLImageKernel>
create_pyramid_transform_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, uint32_t buf_index, bool is_uv)
{
    char transform_option[1024];
    snprintf (
        transform_option, sizeof(transform_option),
        "-DPYRAMID_UV=%d -DCL_PYRAMID_ENABLE_DUMP=%d", (is_uv ? 1 : 0), CL_PYRAMID_ENABLE_DUMP);

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLPyramidTransformKernel (context, blender, layer, buf_index, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelPyramidTransform], transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_lap_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, uint32_t buf_index, bool is_uv)
{
    char transform_option[1024];
    snprintf (
        transform_option, sizeof(transform_option),
        "-DPYRAMID_UV=%d -DCL_PYRAMID_ENABLE_DUMP=%d", (is_uv ? 1 : 0), CL_PYRAMID_ENABLE_DUMP);

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLPyramidLapKernel (context, blender, layer, buf_index, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelPyramidLap], transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_reconstruct_kernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    bool is_uv)
{
    char transform_option[1024];
    snprintf (
        transform_option, sizeof(transform_option),
        "-DPYRAMID_UV=%d -DCL_PYRAMID_ENABLE_DUMP=%d", (is_uv ? 1 : 0), CL_PYRAMID_ENABLE_DUMP);

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLPyramidReconstructKernel (context, blender, layer, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelPyramidReconstruct], transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_blend_kernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    bool is_uv,
    bool need_seam)
{
    char transform_option[1024];
    snprintf (
        transform_option, sizeof(transform_option),
        "-DPYRAMID_UV=%d -DCL_PYRAMID_ENABLE_DUMP=%d", (is_uv ? 1 : 0), CL_PYRAMID_ENABLE_DUMP);

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLPyramidBlendKernel (context, blender, layer, is_uv, need_seam);
    uint32_t index = KernelPyramidBlender;
    if (need_seam)
        index = KernelSeamBlender;

    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[index], transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_blender_local_scale_kernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    bool is_uv)
{
    char transform_option[1024];
    snprintf (transform_option, sizeof(transform_option), "-DPYRAMID_UV=%d", is_uv ? 1 : 0);

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLBlenderLocalScaleKernel (context, blender, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelPyramidScale], transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid blender local scaling kernel(%s) failed", is_uv ? "UV" : "Y");
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_copy_kernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t buf_index,
    bool is_uv)
{
    char transform_option[1024];
    snprintf (transform_option, sizeof(transform_option), "-DPYRAMID_UV=%d", (is_uv ? 1 : 0));

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLPyramidCopyKernel (context, blender, buf_index, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelPyramidCopy], transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load pyramid blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_seam_diff_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender)
{
    SmartPtr<CLImageKernel> kernel;
    kernel = new CLSeamDiffKernel (context, blender);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelImageDiff], NULL) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load seam diff kernel failed");
    return kernel;
}

static SmartPtr<CLImageKernel>
create_seam_DP_kernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender)
{
    SmartPtr<CLImageKernel> kernel;
    kernel = new CLSeamDPKernel (context, blender);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelSeamDP], NULL) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load seam DP kernel failed");
    return kernel;
}

static SmartPtr<CLImageKernel>
create_seam_mask_scale_kernel (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    bool need_scale,
    bool need_slm)
{
    char build_option[1024];
    snprintf (build_option, sizeof(build_option), "-DENABLE_MASK_GAUSS_SCALE=%d", (need_scale ? 1 : 0));
    int kernel_idx = (need_slm ? KernelSeamMaskScaleSLM : KernelSeamMaskScale);

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLPyramidSeamMaskKernel (context, blender, layer, need_scale, need_slm);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[kernel_idx], build_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load seam mask scale kernel failed");
    return kernel;
}

SmartPtr<CLImageHandler>
create_pyramid_blender (
    const SmartPtr<CLContext> &context, int layer, bool need_uv,
    bool need_seam, CLBlenderScaleMode scale_mode)
{
    SmartPtr<CLPyramidBlender> blender;
    SmartPtr<CLImageKernel> kernel;
    int i = 0;
    uint32_t buf_index = 0;
    int max_plane = (need_uv ? 2 : 1);
    bool uv_status[2] = {false, true};

    XCAM_FAIL_RETURN (
        ERROR,
        layer > 0 && layer <= XCAM_CL_PYRAMID_MAX_LEVEL,
        NULL,
        "create_pyramid_blender failed with wrong layer:%d, please set it between %d and %d",
        layer, 1, XCAM_CL_PYRAMID_MAX_LEVEL);

    blender = new CLPyramidBlender (context, "cl_pyramid_blender", layer, need_uv, need_seam, scale_mode);
    XCAM_ASSERT (blender.ptr ());

    if (need_seam) {
        kernel = create_seam_diff_kernel (context, blender);
        XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create seam diff kernel failed");
        blender->add_kernel (kernel);

        kernel = create_seam_DP_kernel (context, blender);
        XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create seam DP kernel failed");
        blender->add_kernel (kernel);

        for (i = 0; i < layer; ++i) {
            bool need_scale = (i < layer - 1);
            bool need_slm = (i == 0);
            kernel = create_seam_mask_scale_kernel (context, blender, (uint32_t)i, need_scale, need_slm);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create seam mask scale kernel failed");
            blender->add_kernel (kernel);
        }
    }

    for (int plane = 0; plane < max_plane; ++plane) {
        for (buf_index = 0; buf_index < XCAM_BLENDER_IMAGE_NUM; ++buf_index) {
            for (i = 0; i < layer - 1; ++i) {
                kernel = create_pyramid_transform_kernel (context, blender, (uint32_t)i, buf_index, uv_status[plane]);
                XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid transform kernel failed");
                blender->add_kernel (kernel);

                kernel = create_pyramid_lap_kernel (context, blender, (uint32_t)i, buf_index, uv_status[plane]);
                XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid lap transform kernel failed");
                blender->add_kernel (kernel);
            }
        }

        for (i = 0; i < layer; ++i) {
            kernel = create_pyramid_blend_kernel (context, blender, (uint32_t)i, uv_status[plane], need_seam);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid blend kernel failed");
            blender->add_kernel (kernel);
        }

        for (i = layer - 2; i >= 0 && i < layer; --i) {
            kernel = create_pyramid_reconstruct_kernel (context, blender, (uint32_t)i, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid reconstruct kernel failed");
            blender->add_kernel (kernel);
        }

        if (scale_mode == CLBlenderScaleLocal) {
            kernel = create_pyramid_blender_local_scale_kernel (context, blender, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid blender local scaling kernel failed");
            blender->add_kernel (kernel);
        }

        for (buf_index = 0; buf_index < XCAM_BLENDER_IMAGE_NUM; ++buf_index) {
            kernel = create_pyramid_copy_kernel (context, blender, buf_index, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid copy kernel failed");
            blender->add_kernel (kernel);
        }
    }

    return blender;
}

}
