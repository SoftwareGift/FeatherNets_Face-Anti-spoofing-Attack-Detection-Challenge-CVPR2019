/*
 * cl_retinex_handler.h - CL retinex handler.
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
 * Author: wangfei <feix.w.wang@intel.com>
 *             Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CL_RETINEX_HANLDER_H
#define XCAM_CL_RETINEX_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"
#include "x3a_stats_pool.h"
#include "cl_image_scaler.h"
#include "cl_gauss_handler.h"

#define XCAM_RETINEX_MAX_SCALE 1
#define XCAM_RETINEX_SCALER_FACTOR 0.4

namespace XCam {

typedef struct {
    float           gain;
    float           threshold;
    float           log_min;
    float           log_max;
    float           width;
    float           height;
} CLRetinexConfig;

class CLRetinexImageHandler;

class CLRetinexScalerImageKernel
    : public CLScalerKernel
{
public:
    explicit CLRetinexScalerImageKernel (
        SmartPtr<CLContext> &context, CLImageScalerMemoryLayout mem_layout, SmartPtr<CLRetinexImageHandler> &retinex);
    virtual void pre_stop ();

protected:
    //derived from CLScalerKernel
    virtual SmartPtr<DrmBoBuffer> get_output_parameter (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLRetinexScalerImageKernel);
    SmartPtr<CLRetinexImageHandler> _retinex;

};

class CLRetinexGaussImageKernel
    : public CLGaussImageKernel
{
public:
    explicit CLRetinexGaussImageKernel (
        SmartPtr<CLContext> &context,
        SmartPtr<CLRetinexImageHandler> &retinex,
        uint32_t index,
        uint32_t radius, float sigma);
    virtual SmartPtr<DrmBoBuffer> get_input_parameter (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual SmartPtr<DrmBoBuffer> get_output_parameter (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);


private:
    XCAM_DEAD_COPY (CLRetinexGaussImageKernel);

    SmartPtr<CLRetinexImageHandler> _retinex;
    uint32_t                        _index;

};

class CLRetinexImageKernel
    : public CLImageKernel
{
public:
    explicit CLRetinexImageKernel (SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> &retinex);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLRetinexImageKernel);

    SmartPtr<CLImage>                _image_in_ga[XCAM_RETINEX_MAX_SCALE];
    SmartPtr<CLImage>                _image_in_uv;
    SmartPtr<CLImage>                _image_out_uv;
    SmartPtr<CLRetinexImageHandler>  _retinex;
    CLRetinexConfig                  _retinex_config;
};

class CLRetinexImageHandler
    : public CLImageHandler
{
public:
    explicit CLRetinexImageHandler (const char *name);
    bool set_retinex_kernel(SmartPtr<CLRetinexImageKernel> &kernel);
    bool set_retinex_scaler_kernel(SmartPtr<CLRetinexScalerImageKernel> &kernel);
    //bool set_retinex_gauss_kernel(SmartPtr<CLRetinexGaussImageKernel> &kernel);
    SmartPtr<DrmBoBuffer> &get_scaler_buf1 () {
        return _scaler_buf1;
    };
    SmartPtr<DrmBoBuffer> &get_gaussian_buf (uint index) {
        XCAM_ASSERT (index < XCAM_RETINEX_MAX_SCALE);
        return _gaussian_buf[index];
    };

    void pre_stop ();

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    XCamReturn prepare_scaler_buf (const VideoBufferInfo &video_info);

private:
    XCAM_DEAD_COPY (CLRetinexImageHandler);
    SmartPtr<CLRetinexImageKernel>        _retinex_kernel;
    SmartPtr<CLRetinexScalerImageKernel>  _retinex_scaler_kernel;
    //SmartPtr<CLRetinexGaussImageKernel>   _retinex_gauss_kernel;

    double                                _scaler_factor;
    SmartPtr<DrmBoBufferPool>             _scaler_buf_pool;
    SmartPtr<DrmBoBuffer>                 _scaler_buf1;
    SmartPtr<DrmBoBuffer>                 _gaussian_buf[XCAM_RETINEX_MAX_SCALE];

};

SmartPtr<CLImageHandler>
create_cl_retinex_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_RETINEX_HANLDER_H
