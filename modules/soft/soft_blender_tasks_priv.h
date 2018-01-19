/*
 * soft_blender_tasks_priv.h - soft blender tasks private class
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

#ifndef XCAM_SOFT_BLENDER_TASKS_PRIV_H
#define XCAM_SOFT_BLENDER_TASKS_PRIV_H

#include <xcam_std.h>
#include <soft/soft_worker.h>
#include <soft/soft_image.h>
#include <soft/soft_blender.h>

#define SOFT_BLENDER_ALIGNMENT_X 8
#define SOFT_BLENDER_ALIGNMENT_Y 4

#define GAUSS_DOWN_SCALE_RADIUS 2
#define GAUSS_DOWN_SCALE_SIZE  ((GAUSS_DOWN_SCALE_RADIUS)*2+1)

namespace XCam {

namespace XCamSoftTasks {

class GaussScaleGray
    : public SoftWorker
{
public:
    struct Args : SoftArgs {
        SmartPtr<UcharImage>           in_luma, out_luma;
    };

public:
    explicit GaussScaleGray (const char *name = "GaussScaleGray", const SmartPtr<Worker::Callback> &cb = NULL)
        : SoftWorker (name, cb)
    {
        set_work_uint (2, 2);
    }

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);

protected:
    void gauss_luma_2x2 (
        UcharImage *in_luma, UcharImage *out_luma, uint32_t x, uint32_t y);

    inline void multiply_coeff_y (float *out, const float *in, float coef) {
        out[0] += in[0] * coef;
        out[1] += in[1] * coef;
        out[2] += in[2] * coef;
        out[3] += in[3] * coef;
        out[4] += in[4] * coef;
        out[5] += in[5] * coef;
        out[6] += in[6] * coef;
        //out[7] += in[7] * coef;
    }

    template<typename T>
    inline T gauss_sum (const T *input) {
        return (input[0] * coeffs[0] + input[1] * coeffs[1] + input[2] * coeffs[2] +
                input[3] * coeffs[3] + input[4] * coeffs[4]);
    }

protected:
    static const float coeffs[GAUSS_DOWN_SCALE_SIZE];
};

class GaussDownScale
    : public GaussScaleGray
{
public:
    struct Args : GaussScaleGray::Args {
        SmartPtr<Uchar2Image>          in_uv, out_uv;
        const uint32_t                 level;
        const SoftBlender::BufIdx      idx;

        SmartPtr<VideoBuffer>          in_buf;
        SmartPtr<VideoBuffer>          out_buf;

        Args (
            const SmartPtr<ImageHandler::Parameters> &param,
            const uint32_t l, const SoftBlender::BufIdx i,
            const SmartPtr<VideoBuffer> &in,
            const SmartPtr<VideoBuffer> &out)
            : level (l)
            , idx (i)
            , in_buf (in)
            , out_buf (out)
        {
            set_param (param);
        }
    };

public:
    explicit GaussDownScale (const SmartPtr<Worker::Callback> &cb)
        : GaussScaleGray ("GaussDownScale", cb)
    {}

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);

    inline void multiply_coeff_uv (Float2 *out, Float2 *in, float coef) {
        out[0] += in[0] * coef;
        out[1] += in[1] * coef;
        out[2] += in[2] * coef;
        out[3] += in[3] * coef;
        out[4] += in[4] * coef;
    }
};

class BlendTask
    : public SoftWorker
{
public:
    struct Args : SoftArgs {
        SmartPtr<UcharImage>   in_luma[2], out_luma;
        SmartPtr<Uchar2Image>  in_uv[2], out_uv;
        SmartPtr<UcharImage>   mask;

        SmartPtr<VideoBuffer>  out_buf;

        Args (
            const SmartPtr<ImageHandler::Parameters> &param,
            const SmartPtr<UcharImage> &m,
            const SmartPtr<VideoBuffer> &out = NULL)
            : SoftArgs (param)
            , mask (m)
            , out_buf (out)
        {}
    };

public:
    explicit BlendTask (const SmartPtr<Worker::Callback> &cb)
        : SoftWorker ("SoftBlendTask", cb)
    {
        set_work_uint (8, 2);
    }

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
};

class LaplaceTask
    : public SoftWorker
{
public:
    struct Args : SoftArgs {
        SmartPtr<UcharImage>        orig_luma, gauss_luma, out_luma;
        SmartPtr<Uchar2Image>       orig_uv, gauss_uv, out_uv;
        const uint32_t              level;
        const SoftBlender::BufIdx   idx;

        SmartPtr<VideoBuffer>  out_buf;

        Args (
            const SmartPtr<ImageHandler::Parameters> &param,
            const uint32_t l, const SoftBlender::BufIdx i,
            const SmartPtr<VideoBuffer> &out = NULL)
            : SoftArgs (param)
            , level(l)
            , idx (i)
            , out_buf (out)
        {}
    };

public:
    explicit LaplaceTask (const SmartPtr<Worker::Callback> &cb)
        : SoftWorker ("SoftLaplaceTask", cb)
    {
        set_work_uint (8, 4);
    }

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);

    void interplate_luma_8x2 (
        UcharImage *orig_luma, UcharImage *gauss_luma, UcharImage *out_luma,
        uint32_t out_x, uint32_t out_y);
};

class ReconstructTask
    : public SoftWorker
{
public:
    struct Args : SoftArgs {
        SmartPtr<UcharImage>        gauss_luma, lap_luma[2], out_luma;
        SmartPtr<Uchar2Image>       gauss_uv, lap_uv[2], out_uv;
        SmartPtr<UcharImage>        mask;
        const uint32_t              level;

        SmartPtr<VideoBuffer>  out_buf;

        Args (
            const SmartPtr<ImageHandler::Parameters> &param,
            const uint32_t l,
            const SmartPtr<VideoBuffer> &out = NULL)
            : SoftArgs (param)
            , level(l)
            , out_buf (out)
        {}
    };

public:
    explicit ReconstructTask (const SmartPtr<Worker::Callback> &cb)
        : SoftWorker ("SoftReconstructTask", cb)
    {
        set_work_uint (8, 4);
    }

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
};

}

}

#endif //XCAM_SOFT_BLENDER_TASKS_PRIV_H
