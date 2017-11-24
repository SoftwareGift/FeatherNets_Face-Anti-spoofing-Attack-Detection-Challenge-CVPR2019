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

#include "soft_blender_tasks_priv.h"

namespace XCam {

namespace XCamSoftTasks {

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
    XCAM_ASSERT (in_luma && out_luma);

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
    XCAM_ASSERT (in_luma && in_uv);
    XCAM_ASSERT (out_luma && out_uv);

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
    XCAM_LOG_DEBUG ("GaussDownScale work on range:[x:%d, width:%d, y:%d, height:%d]",
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

    XCAM_ASSERT (in0_luma && in0_uv && in1_luma && in1_uv);
    XCAM_ASSERT (out_luma && out_uv);
    XCAM_ASSERT (mask);

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

    XCAM_LOG_DEBUG ("BlendTask work on range:[x:%d, width:%d, y:%d, height:%d]",
                    range.pos[0], range.pos_len[0], range.pos[1], range.pos_len[1]);

    return XCAM_RETURN_NO_ERROR;
}

static inline void
minus_array_8 (float *orig, float *gauss, Uchar *ret)
{
#define ORG_MINUS_GAUSS(i) ret[i] = convert_to_uchar<float> ((orig[i] - gauss[i]) * 0.5f + 128.0f)
    ORG_MINUS_GAUSS(0);
    ORG_MINUS_GAUSS(1);
    ORG_MINUS_GAUSS(2);
    ORG_MINUS_GAUSS(3);
    ORG_MINUS_GAUSS(4);
    ORG_MINUS_GAUSS(5);
    ORG_MINUS_GAUSS(6);
    ORG_MINUS_GAUSS(7);
}

static inline void
interpolate_luma_int_row_8x1 (UcharImage* image, uint32_t fixed_x, uint32_t fixed_y, float *gauss_v, float* ret)
{
    image->read_array<float, 5> (fixed_x, fixed_y, gauss_v);
    ret[0] = gauss_v[0];
    ret[1] = (gauss_v[0] + gauss_v[1]) * 0.5f;
    ret[2] = gauss_v[1];
    ret[3] = (gauss_v[1] + gauss_v[2]) * 0.5f;
    ret[4] = gauss_v[2];
    ret[5] = (gauss_v[2] + gauss_v[3]) * 0.5f;
    ret[6] = gauss_v[3];
    ret[7] = (gauss_v[3] + gauss_v[4]) * 0.5f;
}

static inline void
interpolate_luma_half_row_8x1 (UcharImage* image, uint32_t fixed_x, uint32_t next_y, float *last_gauss_v, float* ret)
{
    float next_gauss_v[5];
    float tmp;
    image->read_array<float, 5> (fixed_x, next_y, next_gauss_v);
    ret[0] = (last_gauss_v[0] + next_gauss_v[0]) / 2.0f;
    ret[2] = (last_gauss_v[1] + next_gauss_v[1]) / 2.0f;
    ret[4] = (last_gauss_v[2] + next_gauss_v[2]) / 2.0f;
    ret[6] = (last_gauss_v[3] + next_gauss_v[3]) / 2.0f;
    tmp = (last_gauss_v[4] + next_gauss_v[4]) / 2.0f;
    ret[1] = (ret[0] + ret[2]) / 2.0f;
    ret[3] = (ret[2] + ret[4]) / 2.0f;
    ret[5] = (ret[4] + ret[6]) / 2.0f;
    ret[7] = (ret[6] + tmp) / 2.0f;
}

void
LaplaceTask::interplate_luma_8x2 (
    UcharImage *orig_luma, UcharImage *gauss_luma, UcharImage *out_luma,
    uint32_t out_x, uint32_t out_y)
{
    uint32_t gauss_x = out_x / 2, first_gauss_y = out_y / 2;
    float inter_value[8];
    float gauss_v[5];
    float orig_v[8];
    Uchar lap_ret[8];
    //interplate instaed of coefficient
    interpolate_luma_int_row_8x1 (gauss_luma, gauss_x, first_gauss_y, gauss_v, inter_value);
    orig_luma->read_array_no_check<float, 8> (out_x, out_y, orig_v);
    minus_array_8 (orig_v, inter_value, lap_ret);
    out_luma->write_array_no_check<8> (out_x, out_y, lap_ret);

    uint32_t next_gauss_y = first_gauss_y + 1;
    interpolate_luma_half_row_8x1 (gauss_luma, gauss_x, next_gauss_y, gauss_v, inter_value);
    orig_luma->read_array_no_check<float, 8> (out_x, out_y + 1, orig_v);
    minus_array_8 (orig_v, inter_value, lap_ret);
    out_luma->write_array_no_check<8> (out_x, out_y + 1, lap_ret);
}

static inline void
minus_array_uv_4 (Float2 *orig, Float2 *gauss, Uchar2 *ret)
{
#define ORG_MINUS_GAUSS_UV(i) orig[i] -= gauss[i]; orig[i] *= 0.5f; orig[i] += 128.0f
    ORG_MINUS_GAUSS_UV(0);
    ORG_MINUS_GAUSS_UV(1);
    ORG_MINUS_GAUSS_UV(2);
    ORG_MINUS_GAUSS_UV(3);
    convert_to_uchar2_N<Float2, 4> (orig, ret);
}

static inline void
interpolate_uv_int_row_4x1 (Uchar2Image *image, uint32_t x, uint32_t y, Float2 *gauss_value, Float2 *ret)
{
    image->read_array<Float2, 3> (x, y, gauss_value);
    ret[0] = gauss_value[0];
    ret[1] = gauss_value[0] + gauss_value[1];
    ret[1] *= 0.5f;
    ret[2] = gauss_value[1];
    ret[3] = gauss_value[1] + gauss_value[2];
    ret[3] *= 0.5f;
}

static inline void
interpolate_uv_half_row_4x1 (Uchar2Image *image, uint32_t x, uint32_t y, Float2 *gauss_value, Float2 *ret)
{
    Float2 next_gauss_uv[3];
    image->read_array<Float2, 3> (x, y, next_gauss_uv);
    ret[0] = (gauss_value[0] + next_gauss_uv[0]) * 0.5f;
    ret[2] = (gauss_value[1] + next_gauss_uv[1]) * 0.5f;
    Float2 tmp = (gauss_value[2] + next_gauss_uv[2]) * 0.5f;
    ret[1] = (ret[0] + ret[2]) * 0.5f;
    ret[3] = (ret[2] + tmp) * 0.5f;
}

XCamReturn
LaplaceTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    SmartPtr<LaplaceTask::Args> args = base.dynamic_cast_ptr<LaplaceTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    UcharImage *orig_luma = args->orig_luma.ptr (), *gauss_luma = args->gauss_luma.ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *orig_uv = args->orig_uv.ptr (), *gauss_uv = args->gauss_uv.ptr (), *out_uv = args->out_uv.ptr ();
    XCAM_ASSERT (orig_luma && orig_uv);
    XCAM_ASSERT (gauss_luma && gauss_uv);
    XCAM_ASSERT (out_luma && out_uv);

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            // 8x4 -pixels each time for luma
            uint32_t out_x = x * 8, out_y = y * 4;
            interplate_luma_8x2 (orig_luma, gauss_luma, out_luma, out_x, out_y);
            interplate_luma_8x2 (orig_luma, gauss_luma, out_luma, out_x, out_y + 2);

            // 4x2 uv
            uint32_t out_uv_x = x * 4, out_uv_y = y * 2;
            uint32_t gauss_uv_x = out_uv_x / 2, gauss_uv_y = out_uv_y / 2;
            Float2 gauss_uv_value[3];
            Float2 orig_uv_value[4];
            Float2 inter_uv_value[4];
            Uchar2 lap_uv_ret[4];
            interpolate_uv_int_row_4x1 (gauss_uv, gauss_uv_x, gauss_uv_y, gauss_uv_value, inter_uv_value);
            orig_uv->read_array_no_check<Float2, 4> (out_uv_x , out_uv_y, orig_uv_value);
            minus_array_uv_4 (orig_uv_value, inter_uv_value, lap_uv_ret);
            out_uv->write_array_no_check<4> (out_uv_x , out_uv_y, lap_uv_ret);

            interpolate_uv_half_row_4x1 (gauss_uv, gauss_uv_x, gauss_uv_y + 1, gauss_uv_value, inter_uv_value);
            orig_uv->read_array_no_check<Float2, 4> (out_uv_x , out_uv_y + 1, orig_uv_value);
            minus_array_uv_4 (orig_uv_value, inter_uv_value, lap_uv_ret);
            out_uv->write_array_no_check<4> (out_uv_x, out_uv_y + 1, lap_uv_ret);
        }
    return XCAM_RETURN_NO_ERROR;
}

static inline void
reconstruct_luma_8x1 (float *lap, float *up_sample, Uchar *result)
{
#define RECONSTRUCT_UP_SAMPLE(i) result[i] = convert_to_uchar<float>(up_sample[i] + lap[i] * 2.0f - 256.0f)
    RECONSTRUCT_UP_SAMPLE(0);
    RECONSTRUCT_UP_SAMPLE(1);
    RECONSTRUCT_UP_SAMPLE(2);
    RECONSTRUCT_UP_SAMPLE(3);
    RECONSTRUCT_UP_SAMPLE(4);
    RECONSTRUCT_UP_SAMPLE(5);
    RECONSTRUCT_UP_SAMPLE(6);
    RECONSTRUCT_UP_SAMPLE(7);
}

static inline void
reconstruct_luma_4x1 (Float2 *lap, Float2 *up_sample, Uchar2 *uv_uc)
{
#define RECONSTRUCT_UP_SAMPLE_UV(i) \
    uv_uc[i].x = convert_to_uchar<float>(up_sample[i].x + lap[i].x * 2.0f - 256.0f); \
    uv_uc[i].y = convert_to_uchar<float>(up_sample[i].y + lap[i].y * 2.0f - 256.0f)

    RECONSTRUCT_UP_SAMPLE_UV (0);
    RECONSTRUCT_UP_SAMPLE_UV (1);
    RECONSTRUCT_UP_SAMPLE_UV (2);
    RECONSTRUCT_UP_SAMPLE_UV (3);
}

XCamReturn
ReconstructTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    SmartPtr<ReconstructTask::Args> args = base.dynamic_cast_ptr<ReconstructTask::Args> ();
    XCAM_ASSERT (args.ptr ());
    UcharImage *lap_luma[2] = {args->lap_luma[0].ptr (), args->lap_luma[1].ptr ()};
    UcharImage *gauss_luma = args->gauss_luma.ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *lap_uv[2] = {args->lap_uv[0].ptr (), args->lap_uv[1].ptr ()};
    Uchar2Image *gauss_uv = args->gauss_uv.ptr (), *out_uv = args->out_uv.ptr ();
    UcharImage *mask_image = args->mask.ptr ();
    XCAM_ASSERT (lap_luma[0] && lap_luma[1] && lap_uv[0] && lap_uv[1]);
    XCAM_ASSERT (gauss_luma && gauss_uv);
    XCAM_ASSERT (out_luma && out_uv);
    XCAM_ASSERT (mask_image);

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            // 8x4 -pixels each time for luma
            float luma_blend[8], luma_mask1[8], luma_mask2[8];
            float luma_sample[8];
            float gauss_data[5];
            Uchar luma_uchar[8];
            uint32_t in_x = x * 8, in_y = y * 4;

            // luma 1st - line
            read_and_blend_pixel_luma_8 (lap_luma[0], lap_luma[1], mask_image, in_x, in_y, luma_blend, luma_mask1);
            interpolate_luma_int_row_8x1 (gauss_luma, in_x / 2, in_y / 2, gauss_data, luma_sample);
            reconstruct_luma_8x1 (luma_blend, luma_sample, luma_uchar);
            out_luma->write_array_no_check<8> (in_x, in_y, luma_uchar);

            // luma 2nd -line
            in_y += 1;
            read_and_blend_pixel_luma_8 (lap_luma[0], lap_luma[1], mask_image, in_x, in_y, luma_blend, luma_mask1);
            interpolate_luma_half_row_8x1 (gauss_luma, in_x / 2, in_y / 2 + 1, gauss_data, luma_sample);
            reconstruct_luma_8x1 (luma_blend, luma_sample, luma_uchar);
            out_luma->write_array_no_check<8> (in_x, in_y, luma_uchar);

            // luma 3rd -line
            in_y += 1;
            read_and_blend_pixel_luma_8 (lap_luma[0], lap_luma[1], mask_image, in_x, in_y, luma_blend, luma_mask2);
            interpolate_luma_int_row_8x1 (gauss_luma, in_x / 2, in_y / 2, gauss_data, luma_sample);
            reconstruct_luma_8x1 (luma_blend, luma_sample, luma_uchar);
            out_luma->write_array_no_check<8> (in_x, in_y, luma_uchar);

            // luma 4th -line
            in_y += 1;
            read_and_blend_pixel_luma_8 (lap_luma[0], lap_luma[1], mask_image, in_x, in_y, luma_blend, luma_mask2);
            interpolate_luma_half_row_8x1 (gauss_luma, in_x / 2, in_y / 2 + 1, gauss_data, luma_sample);
            reconstruct_luma_8x1 (luma_blend, luma_sample, luma_uchar);
            out_luma->write_array_no_check<8> (in_x, in_y, luma_uchar);

            // 4x2-UV process UV
            uint32_t uv_x = x * 4, uv_y = y * 2;
            Float2 uv_blend[4];
            Float2 gauss_uv_value[3];
            Float2 up_sample_uv[4];
            Uchar2 uv_uc[4];
            luma_mask1[1] = luma_mask1[2];
            luma_mask1[2] = luma_mask1[4];
            luma_mask1[3] = luma_mask1[6];
            luma_mask2[1] = luma_mask2[2];
            luma_mask2[2] = luma_mask2[4];
            luma_mask2[3] = luma_mask1[6];

            //1st-line UV
            read_and_blend_uv_4 (lap_uv[0], lap_uv[1], luma_mask1, uv_x, uv_y, uv_blend);
            interpolate_uv_int_row_4x1 (gauss_uv, uv_x / 2, uv_y / 2, gauss_uv_value, up_sample_uv);
            reconstruct_luma_4x1 (uv_blend, up_sample_uv, uv_uc);
            out_uv->write_array_no_check<4> (uv_x, uv_y, uv_uc);

            //2nd-line UV
            uv_y += 1;
            read_and_blend_uv_4 (lap_uv[0], lap_uv[1], luma_mask2, uv_x, uv_y, uv_blend);
            interpolate_uv_half_row_4x1 (gauss_uv, uv_x / 2, uv_y / 2 + 1, gauss_uv_value, up_sample_uv);
            reconstruct_luma_4x1 (uv_blend, up_sample_uv, uv_uc);
            out_uv->write_array_no_check<4> (uv_x, uv_y, uv_uc);
        }
    return XCAM_RETURN_NO_ERROR;
}

}

}
