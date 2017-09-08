/*
 * soft_blender.cpp - soft blender class implementation
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

#include "soft_blender.h"
#include "soft_image.h"
#include "soft_worker.h"

#define SOFT_BLENDER_ALIGNMENT_X 4
#define SOFT_BLENDER_ALIGNMENT_Y 2

#define GAUSS_DOWN_SCALE_RADIUS 2
#define GAUSS_DOWN_SCALE_SIZE  ((GAUSS_DOWN_SCALE_RADIUS)*2+1)

namespace XCam {

DECLARE_WORK_CALLBACK (CbGaussDownScale, SoftBlender, gauss_scale_done);


class GaussDownScale
    : public SoftWorker
{
public:
    struct Args : Arguments {
        SmartPtr<SoftImage<Uchar> >  in_luma, out_luma;
        SmartPtr<SoftImage<Uchar2> >  in_uv, out_uv;
    };

public:
    GaussDownScale ()
        : SoftWorker ("GaussDownScale")
    {}

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
    inline void multiply_coeff_y (float *out, const float *in, float coef) {
        out[0] += in[0] * coef;
        out[1] += in[1] * coef;
        out[2] += in[2] * coef;
        out[3] += in[3] * coef;
        out[4] += in[4] * coef;
        out[5] += in[5] * coef;
        out[6] += in[6] * coef;
        out[7] += in[7] * coef;
    }
    inline void multiply_coeff_uv (Float2 *out, Float2 *in, float coef) {
        out[0] += in[0] * coef;
        out[1] += in[1] * coef;
        out[2] += in[2] * coef;
        out[3] += in[3] * coef;
        out[4] += in[4] * coef;
    }

private:
    static const float coeffs[GAUSS_DOWN_SCALE_SIZE];
};

const float GaussDownScale::coeffs[GAUSS_DOWN_SCALE_SIZE] = {0.152f, 0.222f, 0.252f, 0.222f, 0.152f};


XCamReturn
GaussDownScale::work_range (const SmartPtr<Arguments> &base, const WorkRange &range)
{
    SmartPtr<GaussDownScale::Args> args = base.dynamic_cast_ptr<GaussDownScale::Args> ();
    XCAM_ASSERT (args.ptr ());
    SmartPtr<SoftImage<Uchar> > in_luma = args->in_luma, out_luma = args->out_luma;
    SmartPtr<SoftImage<Uchar2> > in_uv = args->in_uv, out_uv = args->out_uv;

    /*
    * o o o o o o o
    * o o o o o o o
    * o o Y(UV) o Y o o
    * o o o o o o o
    * o o Y o Y o o
    * o o o o o o o
    * o o o o o o o
     */
    for (uint32_t y = range.pos[1]; y < range.pos[1] + range.pos_len[1]; ++y)
        for (uint32_t x = range.pos[0]; x < range.pos[0] + range.pos_len[0]; ++x)
        {
            int32_t in_x = x * 4, in_y = y * 4;
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
            value[0] = sum0[0] * coeffs[0] + sum0[1] * coeffs[1] + sum0[2] * coeffs[2] +
                       sum0[3] * coeffs[3] + sum0[4] * coeffs[4];
            value[1] = sum0[2] * coeffs[0] + sum0[3] * coeffs[1] + sum0[4] * coeffs[2] +
                       sum0[5] * coeffs[3] + sum0[6] * coeffs[4];
            out[0] = convert_to_uchar(value[0]);
            out[1] = convert_to_uchar(value[1]);
            out_luma->write_array_no_check<2> (x * 2, y * 2, out);

            value[0] = sum1[0] * coeffs[0] + sum1[1] * coeffs[1] + sum1[2] * coeffs[2] +
                       sum1[3] * coeffs[3] + sum1[4] * coeffs[4];
            value[1] = sum1[2] * coeffs[0] + sum1[3] * coeffs[1] + sum1[4] * coeffs[2] +
                       sum1[5] * coeffs[3] + sum1[6] * coeffs[4];
            out[0] = convert_to_uchar(value[0]);
            out[1] = convert_to_uchar(value[1]);
            out_luma->write_array_no_check<2> (x * 2, y * 2 + 1, out);

            // calculate UV
            Float2 uv_line[5];
            Float2 uv_sum [5];
            in_x = x * 2, in_y = y * 2;
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
            uv_value = uv_sum[0] * coeffs[0] + uv_sum[1] * coeffs[1] + uv_sum[2] * coeffs[2] +
                       uv_sum[3] * coeffs[3] + uv_sum[4] * coeffs[4];
            Uchar2 uv_out(convert_to_uchar(uv_value.x), convert_to_uchar(uv_value.y));
            out_uv->write_data_no_check (x, y, uv_out);
        }

    //printf ("done\n");
    XCAM_LOG_INFO ("GaussDownScale work on range:[x:%d, width:%d, y:%d, height:%d]",
                   range.pos[0], range.pos_len[0], range.pos[1], range.pos_len[1]);

    return XCAM_RETURN_NO_ERROR;
}

SoftBlender::SoftBlender (const char *name)
    : SoftHandler (name)
    , Blender (SOFT_BLENDER_ALIGNMENT_X, SOFT_BLENDER_ALIGNMENT_Y)
{
}

SoftBlender::~SoftBlender ()
{
}

XCamReturn
SoftBlender::blend (
    const SmartPtr<VideoBuffer> &in0,
    const SmartPtr<VideoBuffer> &in1,
    SmartPtr<VideoBuffer> &out_buf)
{
    in0->attach_buffer (in1);
    SmartPtr<ImageHandler::Parameters> params = new ImageHandler::Parameters(in0, out_buf);
    return execute_buffer (params, true);
}

SmartPtr<Worker::Arguments>
SoftBlender::get_first_worker_args (const SmartPtr<SoftWorker> &worker, SmartPtr<ImageHandler::Parameters> &params)
{
    const VideoBufferInfo &in_info = params->in_buf->get_video_info ();
    const VideoBufferInfo &out_info = params->out_buf->get_video_info ();

    SmartPtr<GaussDownScale::Args> args = new GaussDownScale::Args ();
    args->in_luma = new SoftImage<Uchar>
    (params->in_buf, in_info.width, in_info.height, in_info.strides[0], in_info.offsets[0]);
    args->in_uv = new SoftImage<Uchar2>
    (params->in_buf, in_info.width / 2, in_info.height / 2, in_info.strides[1], in_info.offsets[1]);
    args->out_luma = new SoftImage<Uchar>
    (params->out_buf, out_info.width, out_info.height, out_info.strides[0], out_info.offsets[0]);
    args->out_uv = new SoftImage<Uchar2>
    (params->out_buf, out_info.width / 2, out_info.height / 2, out_info.strides[1], out_info.offsets[1]);

    WorkSize global_size (out_info.width / 2, out_info.height / 2);
    WorkSize local_size (global_size.value[0] / 2 , global_size.value[1] / 2 );
    worker->set_local_size (local_size);
    worker->set_global_size (global_size);
    params->in_buf.release ();

    return args;
};

XCamReturn
SoftBlender::gauss_scale_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_UNUSED (args);
    return last_worker_done (error);
}

SmartPtr<SoftHandler>
create_soft_blender ()
{
    SmartPtr<SoftBlender> blender = new SoftBlender();
    XCAM_ASSERT (blender.ptr ());

    SmartPtr<SoftWorker> gauss_scale = new GaussDownScale ();
    XCAM_ASSERT (gauss_scale.ptr ());
    gauss_scale->set_callback (new CbGaussDownScale(blender));

    XCAM_FAIL_RETURN (
        ERROR, blender->set_first_worker (gauss_scale), NULL,
        "softblender set first worker failed.");

    return blender;
}

SmartPtr<Blender>
Blender::create_soft_blender ()
{
    SmartPtr<SoftHandler> handler = XCam::create_soft_blender ();
    return handler.dynamic_cast_ptr<Blender> ();
}

}
