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

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_memory.h>
#include <stats_callback_interface.h>

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
        const SmartPtr<CLContext> &context, CLImageScalerMemoryLayout mem_layout);

public:
    CLImageScalerMemoryLayout get_mem_layout () const {
        return _mem_layout;
    };

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

    //new virtual functions
    virtual SmartPtr<VideoBuffer> get_input_buffer () = 0;
    virtual SmartPtr<VideoBuffer> get_output_buffer () = 0;

protected:
    CLImageScalerMemoryLayout _mem_layout;
};

class CLImageScalerKernel
    : public CLScalerKernel
{
public:
    explicit CLImageScalerKernel (
        const SmartPtr<CLContext> &context, CLImageScalerMemoryLayout mem_layout, SmartPtr<CLImageScaler> &scaler);

protected:
    virtual SmartPtr<VideoBuffer> get_input_buffer ();
    virtual SmartPtr<VideoBuffer> get_output_buffer ();

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
    explicit CLImageScaler (const SmartPtr<CLContext> &context);
    void set_buffer_callback (SmartPtr<StatsCallback> &callback) {
        _scaler_callback = callback;
    }

    bool set_scaler_factor (const double h_factor, const double v_factor);
    bool get_scaler_factor (double &h_factor, double &v_factor) const;
    SmartPtr<VideoBuffer> &get_scaler_buf () {
        return _scaler_buf;
    };

    void emit_stop ();

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

private:
    XCamReturn prepare_scaler_buf (const VideoBufferInfo &video_info, SmartPtr<VideoBuffer> &output);
    XCamReturn post_buffer (const SmartPtr<VideoBuffer> &buffer);

private:
    double                     _h_scaler_factor;
    double                     _v_scaler_factor;
    SmartPtr<BufferPool>       _scaler_buf_pool;
    SmartPtr<VideoBuffer>      _scaler_buf;
    SmartPtr<StatsCallback>    _scaler_callback;
};

SmartPtr<CLImageHandler>
create_cl_image_scaler_handler (const SmartPtr<CLContext> &context, uint32_t format);

};

#endif // XCAM_CL_IMAGE_SCALER_H
