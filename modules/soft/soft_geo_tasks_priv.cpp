/*
 * soft_geo_tasks_priv.cpp - soft geometry map tasks
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

#include "soft_geo_tasks_priv.h"

namespace XCam {

namespace XCamSoftTasks {

enum BoundState {
    BoundInternal = 0,
    BoundCritical,
    BoundExternal
};

inline void check_bound (const uint32_t &img_w, const uint32_t &img_h, Float2 *in_pos,
    const uint32_t &max_idx, BoundState &bound)
{
    if (in_pos[0].x >= 0.0f && in_pos[max_idx].x >= 0.0f && in_pos[0].x < img_w && in_pos[max_idx].x < img_w &&
            in_pos[0].y >= 0.0f && in_pos[max_idx].y >= 0.0f && in_pos[0].y < img_h && in_pos[max_idx].y < img_h)
        bound = BoundInternal;
    else if ((in_pos[0].x < 0.0f && in_pos[max_idx].x < 0.0f) || (in_pos[0].x >= img_w && in_pos[max_idx].x >= img_w) ||
             (in_pos[0].y < 0.0f && in_pos[max_idx].y < 0.0f) || (in_pos[0].y >= img_h && in_pos[max_idx].y >= img_h))
        bound = BoundExternal;
    else
        bound = BoundCritical;
}

template <typename TypeT>
inline void calc_critical_pixels (const uint32_t &img_w, const uint32_t &img_h, Float2 *in_pos,
    const uint32_t &max_idx, const TypeT &zero_byte, TypeT *luma)
{
    for (uint32_t idx = 0; idx < max_idx; ++idx) {
        if (in_pos[idx].x < 0.0f || in_pos[idx].x >= img_w || in_pos[idx].y < 0.0f || in_pos[idx].y >= img_h)
            luma[idx] = zero_byte;
    }
}

static void map_image (
    const UcharImage *in_luma, const Uchar2Image *in_uv,
    UcharImage *out_luma, Uchar2Image *out_uv, const Float2Image *lut,
    const uint32_t &luma_w, const uint32_t &luma_h, const uint32_t &uv_w, const uint32_t &uv_h,
    const uint32_t &x_idx, const uint32_t &y_idx, const uint32_t &out_x, const uint32_t &out_y,
    const Float2 &first, const Float2 &step, const Uchar *zero_luma_byte, const Uchar2 *zero_uv_byte)
{
    Float2 lut_pos[8] = {
        first, Float2(first.x + step.x, first.y),
        Float2(first.x + step.x * 2, first.y), Float2(first.x + step.x * 3, first.y),
        Float2(first.x + step.x * 4, first.y), Float2(first.x + step.x * 5, first.y),
        Float2(first.x + step.x * 6, first.y), Float2(first.x + step.x * 7, first.y)
    };

    //1st-line luma
    Float2 in_pos[8];
    float  luma_value[8];
    Uchar  luma_uc[8];
    BoundState bound = BoundInternal;
    lut->read_interpolate_array<Float2, 8> (lut_pos, in_pos);
    check_bound (luma_w, luma_h, in_pos, 7, bound);
    if (bound == BoundExternal)
        out_luma->write_array_no_check<8> (out_x, out_y, zero_luma_byte);
    else {
        in_luma->read_interpolate_array<float, 8> (in_pos, luma_value);
        convert_to_uchar_N<float, 8> (luma_value, luma_uc);
        if (bound == BoundCritical)
            calc_critical_pixels (luma_w, luma_h, in_pos, 8, zero_luma_byte[0], luma_uc);
        out_luma->write_array_no_check<8> (out_x, out_y, luma_uc);
    }

    //4x1 UV
    Float2 uv_value[4];
    Uchar2 uv_uc[4];
    in_pos[0] /= 2.0f;
    in_pos[1] = in_pos[2] / 2.0f;
    in_pos[2] = in_pos[4] / 2.0f;
    in_pos[3] = in_pos[6] / 2.0f;
    check_bound (uv_w, uv_h, in_pos, 3, bound);
    if (bound == BoundExternal)
        out_uv->write_array_no_check<4> (x_idx * 4, y_idx, zero_uv_byte);
    else {
        in_uv->read_interpolate_array<Float2, 4> (in_pos, uv_value);
        convert_to_uchar2_N<Float2, 4> (uv_value, uv_uc);
        if (bound == BoundCritical)
            calc_critical_pixels (uv_w, uv_h, in_pos, 4, zero_uv_byte[0], uv_uc);
        out_uv->write_array_no_check<4> (x_idx * 4, y_idx, uv_uc);
    }

    //2nd-line luma
    lut_pos[0].y = lut_pos[1].y = lut_pos[2].y = lut_pos[3].y = lut_pos[4].y = lut_pos[5].y =
                                  lut_pos[6].y = lut_pos[7].y = first.y + step.y;
    lut->read_interpolate_array<Float2, 8> (lut_pos, in_pos);
    check_bound (luma_w, luma_h, in_pos, 7, bound);
    if (bound == BoundExternal)
        out_luma->write_array_no_check<8> (out_x, out_y + 1, zero_luma_byte);
    else {
        in_luma->read_interpolate_array<float, 8> (in_pos, luma_value);
        convert_to_uchar_N<float, 8> (luma_value, luma_uc);
        if (bound == BoundCritical)
            calc_critical_pixels (luma_w, luma_h, in_pos, 8, zero_luma_byte[0], luma_uc);
        out_luma->write_array_no_check<8> (out_x, out_y + 1, luma_uc);
    }
}

XCamReturn
GeoMapTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    static const Uchar zero_luma_byte[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    static const Uchar2 zero_uv_byte[4] = {{128, 128}, {128, 128}, {128, 128}, {128, 128}};
    SmartPtr<GeoMapTask::Args> args = base.dynamic_cast_ptr<GeoMapTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    UcharImage *in_luma = args->in_luma.ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = args->in_uv.ptr (), *out_uv = args->out_uv.ptr ();
    Float2Image *lut = args->lookup_table.ptr ();
    XCAM_ASSERT (in_luma && in_uv);
    XCAM_ASSERT (out_luma && out_uv);
    XCAM_ASSERT (lut);

    Float2 factors = args->factors;
    XCAM_ASSERT (!XCAM_DOUBLE_EQUAL_AROUND (factors.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factors.y, 0.0f));

    Float2 step = Float2(1.0f, 1.0f) / factors;

    Float2 out_center ((out_luma->get_width () - 1.0f ) / 2.0f, (out_luma->get_height () - 1.0f ) / 2.0f);
    Float2 lut_center ((lut->get_width () - 1.0f) / 2.0f, (lut->get_height () - 1.0f) / 2.0f);

    uint32_t luma_w = in_luma->get_width ();
    uint32_t luma_h = in_luma->get_height ();
    uint32_t uv_w = in_uv->get_width ();
    uint32_t uv_h = in_uv->get_height ();

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            uint32_t out_x = x * 8, out_y = y * 2;

            // calculate 8x2 luma, center aligned
            Float2 out_pos (out_x, out_y);
            out_pos -= out_center;
            Float2 first = out_pos / factors;
            first += lut_center;

            map_image (in_luma, in_uv, out_luma, out_uv, lut, luma_w, luma_h, uv_w, uv_h,
                       x, y, out_x, out_y, first, step, zero_luma_byte, zero_uv_byte);
        }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GeoMapDualConstTask::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    static const Uchar zero_luma_byte[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    static const Uchar2 zero_uv_byte[4] = {{128, 128}, {128, 128}, {128, 128}, {128, 128}};
    SmartPtr<GeoMapDualConstTask::Args> args = base.dynamic_cast_ptr<GeoMapDualConstTask::Args> ();
    XCAM_ASSERT (args.ptr ());

    UcharImage *in_luma = args->in_luma.ptr (), *out_luma = args->out_luma.ptr ();
    Uchar2Image *in_uv = args->in_uv.ptr (), *out_uv = args->out_uv.ptr ();
    Float2Image *lut = args->lookup_table.ptr ();
    XCAM_ASSERT (in_luma && in_uv);
    XCAM_ASSERT (out_luma && out_uv);
    XCAM_ASSERT (lut);

    Float2 left_factor = args->left_factor;
    Float2 right_factor = args->right_factor;
    XCAM_ASSERT (
        !XCAM_DOUBLE_EQUAL_AROUND (left_factor.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (left_factor.y, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (right_factor.x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (right_factor.y, 0.0f));

    Float2 left_step = Float2(1.0f, 1.0f) / left_factor;
    Float2 right_step = Float2(1.0f, 1.0f) / right_factor;

    Float2 out_center ((out_luma->get_width () - 1.0f ) / 2.0f, (out_luma->get_height () - 1.0f ) / 2.0f);
    Float2 lut_center ((lut->get_width () - 1.0f) / 2.0f, (lut->get_height () - 1.0f) / 2.0f);

    uint32_t luma_w = in_luma->get_width ();
    uint32_t luma_h = in_luma->get_height ();
    uint32_t uv_w = in_uv->get_width ();
    uint32_t uv_h = in_uv->get_height ();

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            uint32_t out_x = x * 8, out_y = y * 2;
            Float2 &factor = (out_x + 4 < out_center.x) ? left_factor : right_factor;
            Float2 &step = (out_x + 4 < out_center.x) ? left_step : right_step;

            // calculate 8x2 luma, center aligned
            Float2 out_pos (out_x, out_y);
            out_pos -= out_center;
            Float2 first = out_pos / factor;
            first += lut_center;

            map_image (in_luma, in_uv, out_luma, out_uv, lut, luma_w, luma_h, uv_w, uv_h,
                       x, y, out_x, out_y, first, step, zero_luma_byte, zero_uv_byte);
        }

    return XCAM_RETURN_NO_ERROR;
}

}

}
