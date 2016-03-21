/*
 * cl_image_scaler.h - CL image scaler
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_CL_IMAGE_SCALER_H
#define XCAM_CL_IMAGE_SCALER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "cl_memory.h"
#include "scaled_buffer_pool.h"
#include "stats_callback_interface.h"

namespace XCam {

enum CLImageScalerMemoryLayout {
    CL_IMAGE_SCALER_NV12_Y = 0,
    CL_IMAGE_SCALER_NV12_UV = 1,
    CL_IMAGE_SCALER_RGBA = 2,
};

#define XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE0 8
#define XCAM_CL_IMAGE_SCALER_KERNEL_LOCAL_WORK_SIZE1 4

class CLImageScaler;

class CLScalerKernel
    : public CLImageKernel
{
public:
    explicit CLScalerKernel (
        SmartPtr<CLContext> &context, CLImageScalerMemoryLayout mem_layout);

public:
    CLImageScalerMemoryLayout get_mem_layout () const {
        return _mem_layout;
    };
    uint32_t get_pixel_format () const {
        return _pixel_format;
    };

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

    //new virtual functions
    virtual SmartPtr<DrmBoBuffer> get_input_parameter (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    virtual SmartPtr<DrmBoBuffer> get_output_parameter (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLScalerKernel);

protected:
    uint32_t _pixel_format;
    CLImageScalerMemoryLayout _mem_layout;
    uint32_t _output_width;
    uint32_t _output_height;
    SmartPtr<CLImage> _cl_image_out;
};

class CLImageScalerKernel
    : public CLScalerKernel
{
public:
    explicit CLImageScalerKernel (
        SmartPtr<CLContext> &context, CLImageScalerMemoryLayout mem_layout, SmartPtr<CLImageScaler> &scaler);

protected:
    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
    virtual void pre_stop ();

    virtual SmartPtr<DrmBoBuffer> get_output_parameter (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLImageScalerKernel);

private:
    SmartPtr<CLImageScaler> _scaler;
};

class CLImageScaler
    : public CLImageHandler
{
    friend class CLImageScalerKernel;
public:
    explicit CLImageScaler ();
    void set_buffer_callback (SmartPtr<StatsCallback> &callback) {
        _scaler_callback = callback;
    }

    bool set_scaler_factor (const double factor);
    double get_scaler_factor () const {
        return _scaler_factor;
    };
    SmartPtr<DrmBoBuffer> &get_scaler_buf () {
        return _scaler_buf;
    };

    void pre_stop ();

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    XCamReturn prepare_scaler_buf (const VideoBufferInfo &video_info, SmartPtr<DrmBoBuffer> &output);

private:
    XCamReturn post_buffer (const SmartPtr<ScaledVideoBuffer> &buffer);
    XCAM_DEAD_COPY (CLImageScaler);

private:
    double _scaler_factor;
    SmartPtr<ScaledVideoBufferPool> _scaler_buf_pool;
    SmartPtr<DrmBoBuffer>   _scaler_buf;
    SmartPtr<StatsCallback> _scaler_callback;
};

SmartPtr<CLImageHandler>
create_cl_image_scaler_handler (SmartPtr<CLContext> &context, uint32_t format);

};

#endif // XCAM_CL_IMAGE_SCALER_H
