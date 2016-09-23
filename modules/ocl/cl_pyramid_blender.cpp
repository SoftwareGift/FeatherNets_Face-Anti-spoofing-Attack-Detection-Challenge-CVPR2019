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
#include "cl_device.h"
#include "cl_image_bo_buffer.h"
#include "cl_utils.h"

//#define SAMPLER_POSITION_OFFSET -0.25f
#define SAMPLER_POSITION_OFFSET 0.0f

namespace XCam {

enum {
    KernelPyramidTransform   = 0,
    KernelPyramidReconstruct,
    KernelPyramidBlender,
    KernelPyramidCopy,
    KernelPyramidLap,
};

const XCamKernelInfo kernels_info [] = {
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
        "kernel_pyramid_copy",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
    {
        "kernel_lap_transform",
#include "kernel_gauss_lap_pyramid.clx"
        , 0,
    },
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
        for (int i = 0; i < XCAM_CL_BLENDER_IMAGE_NUM; ++i) {
            gauss_offset_x[plane][i] = 0;
            lap_offset_x[plane][i] = 0;
        }
        mask_width [plane] = 0;
    }
}

CLPyramidBlender::CLPyramidBlender (const char *name, int layers, bool need_uv)
    : CLBlender (name, need_uv)
    , _layers (0)
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
    XCAM_ASSERT (buf_index < XCAM_CL_BLENDER_IMAGE_NUM);
    uint32_t plane = (is_uv ? 1 : 0);
    return _pyramid_layers[layer].gauss_image[plane][buf_index];
}

SmartPtr<CLImage>
CLPyramidBlender::get_lap_image (uint32_t layer, uint32_t buf_index, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    XCAM_ASSERT (buf_index < XCAM_CL_BLENDER_IMAGE_NUM);
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

SmartPtr<CLBuffer>
CLPyramidBlender::get_blend_mask (uint32_t layer, bool is_uv)
{
    XCAM_ASSERT (layer < _layers);
    uint32_t plane = (is_uv ? 1 : 0);
    return _pyramid_layers[layer].blend_mask[plane];
}

const PyramidLayer &
CLPyramidBlender::get_pyramid_layer (uint32_t layer) const
{
    return _pyramid_layers[layer];
}

void
PyramidLayer::bind_buf_to_layer0 (
    SmartPtr<CLContext> context,
    SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1, SmartPtr<DrmBoBuffer> &output,
    const Rect &merge0_rect, const Rect &merge1_rect, bool need_uv)
{
    const VideoBufferInfo &in0_info = input0->get_video_info ();
    const VideoBufferInfo &in1_info = input1->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    int max_plane = (need_uv ? 2 : 1);
    uint32_t divider_vert[2] = {1, 2};

    XCAM_ASSERT (in0_info.height == in1_info.height);
    XCAM_ASSERT (in0_info.width + in1_info.width >= out_info.width);
    //XCAM_ASSERT (out_info.height == in0_info.height);
    XCAM_ASSERT (merge0_rect.width == merge1_rect.width);

    this->blend_width = XCAM_ALIGN_UP (merge0_rect.width, XCAM_BLENDER_ALIGNED_WIDTH);
    this->blend_height = merge0_rect.height;

    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
    cl_desc.format.image_channel_order = CL_RGBA;

    for (int i_plane = 0; i_plane < max_plane; ++i_plane) {
        cl_desc.width = in0_info.width / 8;
        cl_desc.height = in0_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = in0_info.strides[i_plane];
        this->gauss_image[i_plane][0] = new CLVaImage (context, input0, cl_desc, in0_info.offsets[i_plane]);
        this->gauss_offset_x[i_plane][0] = merge0_rect.pos_x; // input0 offset

        cl_desc.width = in1_info.width / 8;
        cl_desc.height = in1_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = in1_info.strides[i_plane];
        this->gauss_image[i_plane][1] = new CLVaImage (context, input1, cl_desc, in1_info.offsets[i_plane]);
        this->gauss_offset_x[i_plane][1] = merge1_rect.pos_x; // input1 offset

        cl_desc.width = out_info.width / 8;
        cl_desc.height = out_info.height / divider_vert[i_plane];
        cl_desc.row_pitch = out_info.strides[i_plane];

        this->blend_image[i_plane][ReconstructImageIndex] = new CLVaImage (context, output, cl_desc, out_info.offsets[i_plane]);
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
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);
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
        for (int i_image = 0; i_image < XCAM_CL_BLENDER_IMAGE_NUM; ++i_image) {
            cl_desc_set.row_pitch = 0;
            cl_desc_set.width = XCAM_ALIGN_UP (this->blend_width, XCAM_BLENDER_ALIGNED_WIDTH) / 8;
            cl_desc_set.height = XCAM_ALIGN_UP (this->blend_height, divider_vert[plane]) / divider_vert[plane];

            //gauss y image created by cl buffer
            row_pitch = CLImage::calculate_pixel_bytes (cl_desc_set.format) * cl_desc_set.width;
            size = row_pitch * cl_desc_set.height;
            cl_buf = new CLBuffer (context, size);
            XCAM_ASSERT (cl_buf.ptr () && cl_buf->is_valid ());
            cl_desc_set.row_pitch = row_pitch;
            this->gauss_image[plane][i_image] = new CLImage2D (context, cl_desc_set, 0, cl_buf);
            XCAM_ASSERT (this->gauss_image[plane][i_image].ptr ());
            this->gauss_offset_x[plane][i_image]  = 0; // offset to 0, need recalculate if for deep multi-band blender
        }

        cl_desc_set.width = XCAM_ALIGN_UP (this->blend_width, XCAM_BLENDER_ALIGNED_WIDTH) / 8;
        cl_desc_set.height = XCAM_ALIGN_UP (this->blend_height, divider_vert[plane]) / divider_vert[plane];
        row_pitch = CLImage::calculate_pixel_bytes (cl_desc_set.format) * cl_desc_set.width;
        size = row_pitch * cl_desc_set.height;
        cl_buf = new CLBuffer (context, size);
        XCAM_ASSERT (cl_buf.ptr () && cl_buf->is_valid ());
        cl_desc_set.row_pitch = row_pitch;
        this->blend_image[plane][ReconstructImageIndex] = new CLImage2D (context, cl_desc_set, 0, cl_buf);
        XCAM_ASSERT (this->blend_image[plane][ReconstructImageIndex].ptr ());
        if (!last_layer) {
            cl_desc_set.row_pitch = 0;
            this->blend_image[plane][BlendImageIndex] = new CLImage2D (context, cl_desc_set);
            XCAM_ASSERT (this->blend_image[plane][BlendImageIndex].ptr ());
            for (int i_image = 0; i_image < XCAM_CL_BLENDER_IMAGE_NUM; ++i_image) {
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
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);
    ret = this->blend_mask[0]->enqueue_map((void*&)from_ptr, 0, this->mask_width[0] * sizeof(float));
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);

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

        for (uint32_t i_image = 0; i_image < XCAM_CL_BLENDER_IMAGE_NUM; ++i_image) {
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
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);

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
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);

    ret = prev.blend_mask[0]->enqueue_map((void*&)pre_ptr, 0, prev_size);
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);

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
    SmartPtr<DrmBoBuffer> &input0, SmartPtr<DrmBoBuffer> &input1,
    SmartPtr<DrmBoBuffer> &output)
{
    uint32_t index = 0;
    const Rect & window = get_merge_window ();
    bool need_reallocate = true;

    need_reallocate =
        (window.width != (int32_t)_pyramid_layers[0].blend_width ||
         window.height != (int32_t)_pyramid_layers[0].blend_height);
    _pyramid_layers[0].bind_buf_to_layer0 (
        context, input0, input1, output,
        get_input_merge_area (0), get_input_merge_area (1), need_uv ());

    if (need_reallocate) {
        int g_radius = (((float)(window.width - 1) / 2) / (1 << _layers)) * 1.2f;
        float g_sigma = (float)g_radius;

        _pyramid_layers[0].init_layer0 (context, (0 == _layers - 1), need_uv(), g_radius, g_sigma);

        for (index = 1; index < _layers; ++index) {
            _pyramid_layers[index].blend_width = (_pyramid_layers[index - 1].blend_width + 1) / 2;
            _pyramid_layers[index].blend_height = (_pyramid_layers[index - 1].blend_height + 1) / 2;

            _pyramid_layers[index].build_cl_images (context, (index == _layers - 1), need_uv ());
            gauss_fill_mask (context, _pyramid_layers[index - 1], _pyramid_layers[index], need_uv (), g_radius, g_sigma);
        }
    }

    //last layer buffer redirect
    last_layer_buffer_redirect ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidBlender::execute_done (SmartPtr<DrmBoBuffer> &output)
{
    XCAM_UNUSED (output);
    int max_plane = (need_uv () ? 2 : 1);

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
    SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t layer, bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _is_uv (is_uv)
{
}

XCamReturn
CLPyramidBlendKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    SmartPtr<CLContext> context = get_context ();

    SmartPtr<CLImage> image_in0 = get_input_0 ();
    SmartPtr<CLImage> image_in1 = get_input_1 ();
    SmartPtr<CLImage> image_out = get_ouput ();
    SmartPtr<CLBuffer> buf_mask = get_blend_mask ();
    XCAM_ASSERT (image_in0.ptr () && image_in1.ptr () && image_out.ptr ());
    const CLImageDesc &cl_desc_out = image_out->get_image_desc ();

    arg_count = 0;
    args[arg_count].arg_adress = &image_in0->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_in1->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &buf_mask->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_out->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 8;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out.height, work_size.local[1]);
    return XCAM_RETURN_NO_ERROR;
}

CLPyramidTransformKernel::CLPyramidTransformKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    uint32_t buf_index,
    bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _buf_index (buf_index)
    , _is_uv (is_uv)
    , _gauss_offset_x (0)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
    XCAM_ASSERT (buf_index <= XCAM_CL_BLENDER_IMAGE_NUM);
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
CLPyramidTransformKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
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
    _output_gauss.release ();
    change_image_format (context, image_out_gauss, _output_gauss, cl_desc_out_gauss);
    XCAM_FAIL_RETURN (
        ERROR,
        _output_gauss.ptr () && _output_gauss->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change ouput gauss image format failed");

    _gauss_offset_x = get_input_gauss_offset_x () / 8;
    XCAM_ASSERT (_gauss_offset_x * 8 == get_input_gauss_offset_x ());

    arg_count = 0;
    args[arg_count].arg_adress = &image_in_gauss->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_gauss_offset_x;
    args[arg_count].arg_size = sizeof (_gauss_offset_x);
    ++arg_count;

    args[arg_count].arg_adress = &_output_gauss->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

#if CL_PYRAMID_ENABLE_DUMP
    int plane = _is_uv ? 1 : 0;
    SmartPtr<CLImage> dump_original = _blender->get_pyramid_layer (_layer).dump_original[plane][_buf_index];

    args[arg_count].arg_adress = &dump_original->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    printf ("L%dI%d: gauss_offset_x:%d \n", _layer, _buf_index, _gauss_offset_x);
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

XCamReturn
CLPyramidTransformKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _output_gauss.release ();
    return CLImageKernel::post_execute (output);
}

CLPyramidLapKernel::CLPyramidLapKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    uint32_t buf_index,
    bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _buf_index (buf_index)
    , _is_uv (is_uv)
    , _cur_gauss_offset_x (0)
    , _lap_offset_x (0)
    , _out_width (0)
    , _out_height (0)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
    XCAM_ASSERT (buf_index <= XCAM_CL_BLENDER_IMAGE_NUM);
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
CLPyramidLapKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
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
    _next_gauss.release ();
    change_image_format (context, next_gauss_image_tmp, _next_gauss, cl_desc_next_gauss);
    XCAM_FAIL_RETURN (
        ERROR,
        image_out_lap.ptr () && _next_gauss->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change ouput gauss image format failed");

    next_gauss_pixel_width = cl_desc_next_gauss.width;
    next_gauss_pixel_height = cl_desc_next_gauss.height;

    // out format(current layer): CL_UNSIGNED_INT16 + CL_RGBA
    _out_width = CLImage::calculate_pixel_bytes (cl_desc_next_gauss.format) * cl_desc_next_gauss.width * 2.0f / 8.0f;
    _out_height = next_gauss_pixel_height * 2.0f;
    _sampler_offset_x = SAMPLER_POSITION_OFFSET / next_gauss_pixel_width;
    _sampler_offset_y = SAMPLER_POSITION_OFFSET / next_gauss_pixel_height;

    _cur_gauss_offset_x = get_cur_gauss_offset_x () / 8;
    XCAM_ASSERT (_cur_gauss_offset_x * 8 == get_cur_gauss_offset_x ());
    _lap_offset_x = get_output_lap_offset_x () / 8;
    XCAM_ASSERT (_lap_offset_x * 8 == get_output_lap_offset_x ());

    arg_count = 0;
    args[arg_count].arg_adress = &cur_gauss_image->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_cur_gauss_offset_x;
    args[arg_count].arg_size = sizeof (_cur_gauss_offset_x);
    ++arg_count;

    args[arg_count].arg_adress = &_next_gauss->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_sampler_offset_x;
    args[arg_count].arg_size = sizeof (_sampler_offset_x);
    ++arg_count;
    args[arg_count].arg_adress = &_sampler_offset_y;
    args[arg_count].arg_size = sizeof (_sampler_offset_y);
    ++arg_count;

    args[arg_count].arg_adress = &image_out_lap->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_lap_offset_x;
    args[arg_count].arg_size = sizeof (_lap_offset_x);
    ++arg_count;

    args[arg_count].arg_adress = &_out_width;
    args[arg_count].arg_size = sizeof (_out_width);
    ++arg_count;
    args[arg_count].arg_adress = &_out_height;
    args[arg_count].arg_size = sizeof (_out_height);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (cl_desc_out_lap.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (cl_desc_out_lap.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidLapKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _next_gauss.release ();
    return CLImageKernel::post_execute (output);
}

CLPyramidReconstructKernel::CLPyramidReconstructKernel (
    SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer, bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _layer (layer)
    , _is_uv (is_uv)
    , _in_sampler_offset_x (0.0f)
    , _in_sampler_offset_y (0.0f)
    , _out_reconstruct_width (0.0f)
    , _out_reconstruct_height (0.0f)
    , _out_reconstruct_offset_x (0)
{
    XCAM_ASSERT (layer <= XCAM_CL_PYRAMID_MAX_LEVEL);
}

int
CLPyramidReconstructKernel::get_output_reconstrcut_offset_x ()
{
    if (_layer > 0)
        return 0;
    const Rect & window = _blender->get_merge_window ();
    XCAM_ASSERT (window.pos_x % XCAM_BLENDER_ALIGNED_WIDTH == 0);
    return window.pos_x;
}

XCamReturn
CLPyramidReconstructKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
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
    _input_reconstruct.release ();
    change_image_format (context, image_in_reconst, _input_reconstruct, cl_desc_in_reconst);
    XCAM_FAIL_RETURN (
        ERROR,
        _input_reconstruct.ptr () && _input_reconstruct->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "CLPyramidTransformKernel change ouput gauss image format failed");

    input_gauss_width = cl_desc_in_reconst.width;
    input_gauss_height = cl_desc_in_reconst.height;

    _out_reconstruct_width = CLImage::calculate_pixel_bytes (cl_desc_in_reconst.format) * cl_desc_in_reconst.width * 2.0f / 8.0f;
    _out_reconstruct_height = input_gauss_height * 2.0f;
    _in_sampler_offset_x = SAMPLER_POSITION_OFFSET / input_gauss_width;
    _in_sampler_offset_y = SAMPLER_POSITION_OFFSET / input_gauss_height;

    _out_reconstruct_offset_x = get_output_reconstrcut_offset_x () / 8;
    XCAM_ASSERT (_out_reconstruct_offset_x * 8 == get_output_reconstrcut_offset_x ());

    arg_count = 0;
    args[arg_count].arg_adress = &_input_reconstruct->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_in_sampler_offset_x;
    args[arg_count].arg_size = sizeof (_in_sampler_offset_x);
    ++arg_count;

    args[arg_count].arg_adress = &_in_sampler_offset_y;
    args[arg_count].arg_size = sizeof (_in_sampler_offset_y);
    ++arg_count;

    args[arg_count].arg_adress = &image_in_lap->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &image_out_reconst->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_out_reconstruct_offset_x;
    args[arg_count].arg_size = sizeof (_out_reconstruct_offset_x);
    ++arg_count;

    args[arg_count].arg_adress = &_out_reconstruct_width;
    args[arg_count].arg_size = sizeof (_out_reconstruct_width);
    ++arg_count;

    args[arg_count].arg_adress = &_out_reconstruct_height;
    args[arg_count].arg_size = sizeof (_out_reconstruct_height);
    ++arg_count;

#if CL_PYRAMID_ENABLE_DUMP
    int i_plane = (_is_uv ? 1 : 0);
    const PyramidLayer &cur_layer = _blender->get_pyramid_layer (_layer);
    SmartPtr<CLImage>  dump_gauss_resize = cur_layer.dump_gauss_resize[i_plane];
    SmartPtr<CLImage>  dump_final = cur_layer.dump_final[i_plane];

    args[arg_count].arg_adress = &dump_gauss_resize->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &dump_final->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    printf ("Rec%d: reconstruct_offset_x:%d, out_width:%.2f, out_height:%.2f, in_sampler_offset_x:%.2f, in_sampler_offset_y:%.2f\n",
            _layer, _out_reconstruct_offset_x, _out_reconstruct_width, _out_reconstruct_height,
            _in_sampler_offset_x, _in_sampler_offset_y);
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
        XCamReturn ret = images[i]->enqueue_map ((void *&)ptr[i], origin, region, &row_pitch[i], &slice_pitch[i], CL_MEM_READ_ONLY);
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

    // dump image data to file
    image = this->get_gauss_image (1, 0, false); // layer 1, image 0
    write_image (image, "gauss_L1_I0");

    image = this->get_gauss_image (1, 1, false); // layer 1, image 0
    write_image (image, "gauss_L1_I1");

    image = this->get_lap_image (0, 0, false); // layer 1, image 0
    write_image (image, "lap_L0_I0");

    image = this->get_blend_image (1, false); // layer 1
    write_image (image, "blend_L1");

    image = this->get_blend_image (0, false); // layer 0
    write_image (image, "blend_L0");

#if CL_PYRAMID_ENABLE_DUMP
    image = this->get_pyramid_layer (0).dump_gauss_resize[0];
    write_image (image, "dump_gauss_resize_L0");

    image = this->get_pyramid_layer (0).dump_original[0][0];
    write_image (image, "orginal_L0I0");
    image = this->get_pyramid_layer (0).dump_original[0][1];
    write_image (image, "orginal_L0I1");

    image = this->get_pyramid_layer (0).dump_final[CLBlenderPlaneY];
    write_image (image, "dump_final_L0");
#endif
    this->dump_layer_mask (0, false);
    this->dump_layer_mask (1, false);

    this->dump_layer_mask (0, true);
    this->dump_layer_mask (1, true);

}

XCamReturn
CLPyramidReconstructKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _input_reconstruct.release ();

    return CLImageKernel::post_execute (output);
}


CLPyramidCopyKernel::CLPyramidCopyKernel (
    SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender, uint32_t buf_index, bool is_uv)
    : CLImageKernel (context)
    , _blender (blender)
    , _is_uv (is_uv)
    , _buf_index (buf_index)
    , _in_offset_x (0)
    , _out_offset_x (0)
    , _max_g_x (0)
    , _max_g_y (0)
{
}

XCamReturn
CLPyramidCopyKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    SmartPtr<CLContext> context = get_context ();

    _from = get_input ();
    _to = get_output ();

    const CLImageDesc &to_desc = _to->get_image_desc ();
    const Rect &window = _blender->get_merge_window ();
    const Rect& input_area = _blender->get_input_valid_area (_buf_index);
    const uint32_t input_merge_x = _blender->get_input_merge_area (_buf_index).pos_x;

    if (_buf_index == 0) {
        _in_offset_x = input_area.pos_x / 8;
        _max_g_x = (input_merge_x - input_area.pos_x) / 8;
        _out_offset_x = window.pos_x / 8 - _max_g_x;
    } else {
        _in_offset_x = (input_merge_x + window.width) / 8;
        _out_offset_x = (window.pos_x + window.width) / 8;
        _max_g_x = (input_area.pos_x + input_area.width) / 8 - _in_offset_x;
    }
    _max_g_y = to_desc.height;
    XCAM_ASSERT (_max_g_x > 0 && _max_g_x <= (int)to_desc.width);

#if CL_PYRAMID_ENABLE_DUMP
    printf ("copy(%d), in_offset_x:%d, out_offset_x:%d, max_x:%d\n", _buf_index, _in_offset_x, _out_offset_x, _max_g_x);
#endif

    arg_count = 0;
    args[arg_count].arg_adress = &_from->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_in_offset_x;
    args[arg_count].arg_size = sizeof (_in_offset_x);
    ++arg_count;

    args[arg_count].arg_adress = &_to->get_mem_id ();
    args[arg_count].arg_size = sizeof (cl_mem);
    ++arg_count;

    args[arg_count].arg_adress = &_out_offset_x;
    args[arg_count].arg_size = sizeof (_out_offset_x);
    ++arg_count;

    args[arg_count].arg_adress = &_max_g_x;
    args[arg_count].arg_size = sizeof (_max_g_x);
    ++arg_count;

    args[arg_count].arg_adress = &_max_g_y;
    args[arg_count].arg_size = sizeof (_max_g_y);
    ++arg_count;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (_max_g_x, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (_max_g_y, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLPyramidCopyKernel::post_execute (SmartPtr<DrmBoBuffer> &output)
{
    _from.release ();
    _to.release ();
    return CLImageKernel::post_execute (output);
}

static SmartPtr<CLImageKernel>
create_pyramid_transform_kernel (
    SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
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
        "load linear blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_lap_kernel (
    SmartPtr<CLContext> &context, SmartPtr<CLPyramidBlender> &blender,
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
        "load linear blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}


static SmartPtr<CLImageKernel>
create_pyramid_reconstruct_kernel (
    SmartPtr<CLContext> &context,
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
        "load linear blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_blend_kernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLPyramidBlender> &blender,
    uint32_t layer,
    bool is_uv)
{
    char transform_option[1024];
    snprintf (
        transform_option, sizeof(transform_option),
        "-DPYRAMID_UV=%d -DCL_PYRAMID_ENABLE_DUMP=%d", (is_uv ? 1 : 0), CL_PYRAMID_ENABLE_DUMP);

    SmartPtr<CLImageKernel> kernel;
    kernel = new CLPyramidBlendKernel (context, blender, layer, is_uv);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        kernel->build_kernel (kernels_info[KernelPyramidBlender], transform_option) == XCAM_RETURN_NO_ERROR,
        NULL,
        "load linear blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}

static SmartPtr<CLImageKernel>
create_pyramid_copy_kernel (
    SmartPtr<CLContext> &context,
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
        "load linear blender kernel(%s) failed", (is_uv ? "UV" : "Y"));
    return kernel;
}


SmartPtr<CLImageHandler>
create_pyramid_blender (SmartPtr<CLContext> &context, int layer, bool need_uv)
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

    blender = new CLPyramidBlender ("cl_pyramid_blender", layer, need_uv);
    XCAM_ASSERT (blender.ptr ());

    for (int plane = 0; plane < max_plane; ++plane) {
        for (buf_index = 0; buf_index < XCAM_CL_BLENDER_IMAGE_NUM; ++buf_index) {
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
            kernel = create_pyramid_blend_kernel (context, blender, (uint32_t)i, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid blend kernel failed");
            blender->add_kernel (kernel);
        }

        for (i = layer - 2; i >= 0 && i < layer; --i) {
            kernel = create_pyramid_reconstruct_kernel (context, blender, (uint32_t)i, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid reconstruct kernel failed");
            blender->add_kernel (kernel);
        }

        for (buf_index = 0; buf_index < XCAM_CL_BLENDER_IMAGE_NUM; ++buf_index) {
            kernel = create_pyramid_copy_kernel (context, blender, buf_index, uv_status[plane]);
            XCAM_FAIL_RETURN (ERROR, kernel.ptr (), NULL, "create pyramid copy kernel failed");
            blender->add_kernel (kernel);
        }
    }

    return blender;
}

}
