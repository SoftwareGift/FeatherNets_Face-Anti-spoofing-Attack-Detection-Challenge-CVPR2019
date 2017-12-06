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
    XCAM_ASSERT (
        !XCAM_DOUBLE_EQUAL_AROUND (factors.x, 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (factors.y, 0.0f));

    Float2 out_center ((out_luma->get_width () - 1.0f ) / 2.0f, (out_luma->get_height () - 1.0f ) / 2.0f);
    Float2 lut_center ((lut->get_width () - 1.0f) / 2.0f, (lut->get_height () - 1.0f) / 2.0f);
    float x_step = 1.0f / factors.x;
    float y_step = 1.0f / factors.y;

#undef OUT_BOUND
#define OUT_BOUND(image, first, last) \
        (in_pos[first].x >= image->get_width ()) ||   \
        (in_pos[first].y >= image->get_height ()) ||  \
        (in_pos[last].x <= 0.0f) || (in_pos[last].y <= 0.0f)

    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            // calculate 8x2 luma, center aligned
            Float2 in_pos[8];
            float  luma_value[8];
            Uchar  luma_uc[8];
            uint32_t out_x = x * 8, out_y = y * 2;

            //1st-line luma
            Float2 out_pos (out_x, out_y);
            out_pos -= out_center;
            Float2 first = out_pos / factors;
            first += lut_center;
            Float2 lut_pos[8] = {
                first, Float2(first.x + x_step, first.y),
                Float2(first.x + x_step * 2, first.y), Float2(first.x + x_step * 3, first.y),
                Float2(first.x + x_step * 4, first.y), Float2(first.x + x_step * 5, first.y),
                Float2(first.x + x_step * 6, first.y), Float2(first.x + x_step * 7, first.y)
            };
            lut->read_interpolate_array<Float2, 8> (lut_pos, in_pos);
            in_luma->read_interpolate_array<float, 8> (in_pos, luma_value);
            convert_to_uchar_N<float, 8> (luma_value, luma_uc);
            if (OUT_BOUND (in_luma, 0, 7))
                out_luma->write_array_no_check<8> (out_x, out_y, zero_luma_byte);
            else
                out_luma->write_array_no_check<8> (out_x, out_y, luma_uc);

            //4x1 UV
            Float2  uv_value[4];
            Uchar2  uv_uc[4];
            in_pos[0] /= 2.0f;
            in_pos[1] = in_pos[2] / 2.0f;
            in_pos[2] = in_pos[4] / 2.0f;
            in_pos[3] = in_pos[6] / 2.0f;
            in_uv->read_interpolate_array<Float2, 4> (in_pos, uv_value);
            convert_to_uchar2_N<Float2, 4> (uv_value, uv_uc);
            if (OUT_BOUND (in_uv, 0, 3))
                out_uv->write_array_no_check<4> (x * 4, y, zero_uv_byte);
            else
                out_uv->write_array_no_check<4> (x * 4, y, uv_uc);

            //2nd-line luma
            lut_pos[0].y = lut_pos[1].y = lut_pos[2].y = lut_pos[3].y = lut_pos[4].y = lut_pos[5].y =
                                              lut_pos[6].y = lut_pos[7].y = first.y + y_step;
            lut->read_interpolate_array<Float2, 8> (lut_pos, in_pos);
            in_luma->read_interpolate_array<float, 8> (in_pos, luma_value);
            convert_to_uchar_N<float, 8> (luma_value, luma_uc);
            if (OUT_BOUND (in_luma, 0, 7))
                out_luma->write_array_no_check<8> (out_x, out_y + 1, zero_luma_byte);
            else
                out_luma->write_array_no_check<8> (out_x, out_y + 1, luma_uc);
        }
    return XCAM_RETURN_NO_ERROR;
}

}

}

