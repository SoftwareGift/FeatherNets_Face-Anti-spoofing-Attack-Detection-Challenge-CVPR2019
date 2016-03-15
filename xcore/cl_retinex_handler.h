/*
 * cl_retinex_handler.h - CL retinex handler.
 *
 *  Copyright (c) 2015 Intel Corporation
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
 */

#ifndef XCAM_CL_RETINEX_HANLDER_H
#define XCAM_CL_RETINEX_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"
#include "x3a_stats_pool.h"
#include "cl_image_scaler.h"
#include "cl_gauss_handler.h"

#define XCAM_RETINEX_TABLE_SIZE 5
#define XCAM_RETINEX_SCALE 3
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
    explicit CLRetinexScalerImageKernel (SmartPtr<CLContext> &context, CLImageScalerMemoryLayout mem_layout, SmartPtr<CLRetinexImageHandler> &scaler);
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
    virtual void pre_stop ();

private:
    XCAM_DEAD_COPY (CLRetinexScalerImageKernel);
    SmartPtr<CLRetinexImageHandler> _scaler;

};

class CLRetinexGaussImageKernel
    : public CLGaussImageKernel
{
public:
    explicit CLRetinexGaussImageKernel (SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> &scaler);
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
//    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
//    virtual void pre_stop ();

private:
    XCAM_DEAD_COPY (CLRetinexGaussImageKernel);
    SmartPtr<CLRetinexImageHandler> _scaler;

};

class CLRetinexImageKernel
    : public CLImageKernel
{
public:
    explicit CLRetinexImageKernel (SmartPtr<CLContext> &context, SmartPtr<CLRetinexImageHandler> &scaler);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLRetinexImageKernel);

    SmartPtr<CLImage>                _image_in_ga;
    SmartPtr<CLImage>                _image_in_uv;
    SmartPtr<CLImage>                _image_out_uv;
    SmartPtr<CLRetinexImageHandler>  _scaler;
    CLRetinexConfig                  _retinex_config;
};

class CLRetinexImageHandler
    : public CLImageHandler
{
public:
    explicit CLRetinexImageHandler (const char *name);
    bool set_retinex_kernel(SmartPtr<CLRetinexImageKernel> &kernel);
    bool set_retinex_scaler_kernel(SmartPtr<CLRetinexScalerImageKernel> &kernel);
    bool set_retinex_gauss_kernel(SmartPtr<CLRetinexGaussImageKernel> &kernel);
    SmartPtr<DrmBoBuffer> &get_scaler_buf () {
        return _scaler_buf;
    };
    void pre_stop ();

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    XCamReturn prepare_scaler_buf (const VideoBufferInfo &video_info, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLRetinexImageHandler);
    SmartPtr<CLRetinexImageKernel> _retinex_kernel;
    SmartPtr<CLRetinexScalerImageKernel> _retinex_scaler_kernel;
    SmartPtr<CLRetinexGaussImageKernel> _retinex_gauss_kernel;
    SmartPtr<ScaledVideoBufferPool> _scaler_buf_pool;
    SmartPtr<DrmBoBuffer>   _scaler_buf;
    double _scaler_factor;
};

SmartPtr<CLImageHandler>
create_cl_retinex_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_RETINEX_HANLDER_H
