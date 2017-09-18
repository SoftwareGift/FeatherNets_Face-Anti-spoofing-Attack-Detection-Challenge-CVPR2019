/*
 * soft_blender_tasks_priv.cpp - soft blender tasks private class implementation
 *
 *  Copyright (c) 2017 Intel Corporation
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

#include "xcam_utils.h"
#include "soft_blender_tasks_priv.h"

namespace XCam {

const float GaussScaleGray::coeffs[GAUSS_DOWN_SCALE_SIZE] = {0.152f, 0.222f, 0.252f, 0.222f, 0.152f};

void
GaussScaleGray::gauss_luma_2x2 (
    UcharImage *in_luma, UcharImage *out_luma,
    uint32_t x, uint32_t y)
{
    /*
    * o o o o o o o
    * o o o o o o o
    * o o Y(UV) o Y o o
    * o o o o o o o
    * o o Y o Y o o
    * o o o o o o o
    * o o o o o o o
     */
    uint32_t in_x = x * 4, in_y = y * 4;
    float line[7];
    float sum0[7] = {0.0f};
    float sum1[7] = {0.0f};
    in_luma->read_array<float, 7> (in_x - 2, in_y - 2, line);
    multiply_coeff_y (sum0, line, coeffs[0]);
    in_luma->read_array<float, 7> (in_x - 2, in_y - 1, line);
    multiply_coeff_y (sum0, line, coeffs[1]);
    in_luma->read_array<float, 7> (in_x - 2, in_y, line);
    multiply_coeff_y (sum0, line, coeffs[2]);
    multiply_coeff_y (sum1, line, coeffs[0]);
    in_luma->read_array<float, 7> (in_x - 2, in_y + 1, line);
    multiply_coeff_y (sum0, line, coeffs[3]);
    multiply_coeff_y (sum1, line, coeffs[1]);
    in_luma->read_array<float, 7> (in_x - 2, in_y + 2, line);
    multiply_coeff_y (sum0, line, coeffs[4]);
    multiply_coeff_y (sum1, line, coeffs[2]);
    in_luma->read_array<float, 7> (in_x - 2, in_y + 3, line);
    multiply_coeff_y (sum1, line, coeffs[3]);
    in_luma->read_array<float, 7> (in_x - 2, in_y + 4, line);
    multiply_coeff_y (sum1, line, coeffs[4]);

    float value[2];
    Uchar out[2];
    value[0] = gauss_sum (&sum0[0]);
    value[1] = gauss_sum (&sum0[2]);
    out[0] = convert_to_uchar (value[0]);
    out[1] = convert_to_uchar (value[1]);
    out_luma->write_array_no_check<2> (x * 2, y * 2, out);

    value[0] = gauss_sum (&sum1[0]);
    value[1] = gauss_sum (&sum1[2]);
    out[0] = convert_to_uchar(value[0]);
    out[1] = convert_to_uchar(value[1]);
    out_luma->write_array_no_check<2> (x * 2, y * 2 + 1, out);
}

XCamReturn
GaussScaleGray::work_range (const SmartPtr<Worker::Arguments> &base, const WorkRange &range)
{
    SmartPtr<GaussScaleGray::Args> args = base.dynamic_cast_ptr<GaussScaleGray::Args> ();
    XCAM_ASSERT (args.ptr ());
    UcharImage *in_luma = args->in_luma.ptr (), *out_luma = args->out_luma.ptr ();

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            gauss_luma_2x2 (in_luma, out_luma, x, y);
        }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GaussDownScale::work_range (const SmartPtr<Worker::Arguments> &base, const WorkRange &range)
{
    SmartPtr<GaussDownScale::Args> args = base.dynamic_cast_ptr<GaussDownScale::Args> ();
    XCAM_ASSERT (args.ptr ());
    UcharImage *in_luma = args->in_luma.ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = args->in_uv.ptr (), *out_uv = args->out_uv.ptr ();

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            gauss_luma_2x2 (in_luma, out_luma, x, y);

            // calculate UV
            int32_t in_x = x * 2, in_y = y * 2;
            Float2 uv_line[5];
            Float2 uv_sum [5];

            in_uv->read_array<Float2, 5> (in_x - 2, in_y - 2, uv_line);
            multiply_coeff_uv (uv_sum, uv_line, coeffs[0]);
            in_uv->read_array<Float2, 5> (in_x - 2, in_y - 1, uv_line);
            multiply_coeff_uv (uv_sum, uv_line, coeffs[1]);
            in_uv->read_array<Float2, 5> (in_x - 2, in_y , uv_line);
            multiply_coeff_uv (uv_sum, uv_line, coeffs[2]);
            in_uv->read_array<Float2, 5> (in_x - 2, in_y + 1, uv_line);
            multiply_coeff_uv (uv_sum, uv_line, coeffs[3]);
            in_uv->read_array<Float2, 5> (in_x - 2, in_y + 2, uv_line);
            multiply_coeff_uv (uv_sum, uv_line, coeffs[4]);
            Float2 uv_value;
            uv_value = gauss_sum (&uv_sum[0]);
            Uchar2 uv_out(convert_to_uchar(uv_value.x), convert_to_uchar(uv_value.y));
            out_uv->write_data_no_check (x, y, uv_out);
        }

    //printf ("done\n");
    XCAM_LOG_INFO ("GaussDownScale work on range:[x:%d, width:%d, y:%d, height:%d]",
                   range.pos[0], range.pos_len[0], range.pos[1], range.pos_len[1]);

    return XCAM_RETURN_NO_ERROR;
}

static inline void
blend_luma_8 (const float *luma0, const float *luma1, const float *mask, float *out)
{
    //out[0] = luma0[0] * mask + luma1[0] * ( 1.0f - mask[0]);
#define BLEND_LUMA_8(idx) out[idx] = (luma0[idx] - luma1[idx]) * mask[idx] + luma1[idx]
    BLEND_LUMA_8 (0);
    BLEND_LUMA_8 (1);
    BLEND_LUMA_8 (2);
    BLEND_LUMA_8 (3);
    BLEND_LUMA_8 (4);
    BLEND_LUMA_8 (5);
    BLEND_LUMA_8 (6);
    BLEND_LUMA_8 (7);
}

static inline void
normalize_8 (float *value, const float max)
{
    value[0] /= max;
    value[1] /= max;
    value[2] /= max;
    value[3] /= max;
    value[4] /= max;
    value[5] /= max;
    value[6] /= max;
    value[7] /= max;
}

static inline void
read_and_blend_pixel_luma_8 (
    const UcharImage *in0, const UcharImage *in1,
    const UcharImage *mask,
    const uint32_t in_x, const uint32_t in_y,
    float *out_luma,
    float *out_mask)
{
    float luma0_line[8], luma1_line[8];
    mask->read_array_no_check<float, 8> (in_x, in_y, out_mask);
    in0->read_array_no_check<float, 8> (in_x, in_y, luma0_line);
    in1->read_array_no_check<float, 8> (in_x, in_y, luma1_line);
    normalize_8 (out_mask, 255.0f);
    blend_luma_8 (luma0_line, luma1_line, out_mask, out_luma);
}

static inline void
read_and_blend_uv_4 (
    const Uchar2Image *in_a, const Uchar2Image *in_b,
    const float *mask,
    const uint32_t in_x, const uint32_t in_y,
    Float2 *out_uv)
{
    Float2 line_a[4], line_b[4];
    in_a->read_array_no_check<Float2, 4> (in_x, in_y, line_a);
    in_b->read_array_no_check<Float2, 4> (in_x, in_y, line_b);

    //out_uv[0] = line_a[0] * mask + line_b[0] * ( 1.0f - mask[0]);
#define BLEND_UV_4(i) out_uv[i] = (line_a[i] - line_b[i]) * mask[i] + line_b[i]
    BLEND_UV_4 (0);
    BLEND_UV_4 (1);
    BLEND_UV_4 (2);
    BLEND_UV_4 (3);
}

XCamReturn
BlendTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    SmartPtr<BlendTask::Args> args = base.dynamic_cast_ptr<BlendTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    UcharImage *in0_luma = args->in_luma[0].ptr (), *in1_luma = args->in_luma[1].ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *in0_uv = args->in_uv[0].ptr (), *in1_uv = args->in_uv[1].ptr (), *out_uv = args->out_uv.ptr ();
    UcharImage *mask = args->mask.ptr ();

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            // 8x2 -pixels each time for luma
            uint32_t in_x = x * 8;
            uint32_t in_y = y * 2;
            float luma_blend[8], luma_mask[8];
            Uchar luma_uc[8];

            // process luma (in_x, in_y)
            read_and_blend_pixel_luma_8 (in0_luma, in1_luma, mask, in_x, in_y, luma_blend, luma_mask);
            convert_to_uchar_N<float, 8> (luma_blend, luma_uc);
            out_luma->write_array_no_check<8> (in_x, in_y, luma_uc);

            // process luma (in_x, in_y + 1)
            read_and_blend_pixel_luma_8 (in0_luma, in1_luma, mask, in_x, in_y + 1, luma_blend, luma_mask);
            convert_to_uchar_N<float, 8> (luma_blend, luma_uc);
            out_luma->write_array_no_check<8> (in_x, in_y + 1, luma_uc);

            // process uv(4x1) (uv_x, uv_y)
            uint32_t uv_x = x * 4, uv_y = y;
            Float2 uv_blend[4];
            Uchar2 uv_uc[4];
            luma_mask[1] = luma_mask[2];
            luma_mask[2] = luma_mask[4];
            luma_mask[3] = luma_mask[6];
            read_and_blend_uv_4 (in0_uv, in1_uv, luma_mask, uv_x, uv_y, uv_blend);
            convert_to_uchar2_N<Float2, 4> (uv_blend, uv_uc);
            out_uv->write_array_no_check<4> (uv_x, uv_y, uv_uc);
        }

    XCAM_LOG_INFO ("BlendTask work on range:[x:%d, width:%d, y:%d, height:%d]",
                   range.pos[0], range.pos_len[0], range.pos[1], range.pos_len[1]);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
LaplaceTask::work_range (const SmartPtr<Arguments> &args, const WorkRange &range)
{
    XCAM_UNUSED (args.ptr ());
    XCAM_UNUSED (range);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ReconstructTask::work_range (const SmartPtr<Arguments> &args, const WorkRange &range)
{
    XCAM_UNUSED (args.ptr ());
    XCAM_UNUSED (range);
    return XCAM_RETURN_NO_ERROR;
}

}
