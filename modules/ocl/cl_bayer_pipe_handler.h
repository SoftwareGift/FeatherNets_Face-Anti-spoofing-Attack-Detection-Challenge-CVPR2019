/*
 * cl_bayer_pipe_handler.h - CL bayer pipe handler
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: wangfei <feix.w.wang@intel.com>
 * Author: Shincy Tu <shincy.tu@intel.com>
 */

#ifndef XCAM_CL_BAYER_PIPE_HANDLER_H
#define XCAM_CL_BAYER_PIPE_HANDLER_H

#include <xcam_std.h>
#include <stats_callback_interface.h>
#include <x3a_stats_pool.h>
#include <ocl/cl_context.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_3a_stats_context.h>

#define XCAM_BNR_TABLE_SIZE 64

namespace XCam {

class CLBayerPipeImageHandler;

typedef struct
{
    float           ee_gain;
    float           ee_threshold;
    float           nr_gain;
} CLEeConfig;

class CLBayerPipeImageKernel
    : public CLImageKernel
{
public:
    explicit CLBayerPipeImageKernel (
        const SmartPtr<CLContext> &context,
        SmartPtr<CLBayerPipeImageHandler> &handler);

private:
    SmartPtr<CLBayerPipeImageHandler>     _handler;
};

class CLBayerPipeImageHandler
    : public CLImageHandler
{
    friend class CLBayerPipeImageKernel;

public:
    explicit CLBayerPipeImageHandler (const SmartPtr<CLContext> &context, const char *name);
    bool set_bayer_kernel (SmartPtr<CLBayerPipeImageKernel> &kernel);
    bool set_ee_config (const XCam3aResultEdgeEnhancement &ee);
    bool set_bnr_config (const XCam3aResultBayerNoiseReduction &bnr);
    bool set_output_format (uint32_t fourcc);
    bool enable_denoise (bool enable);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input, VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (
        SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLBayerPipeImageHandler);

private:
    SmartPtr<CLBayerPipeImageKernel>   _bayer_kernel;
    uint32_t                           _output_format;

    uint32_t                           _enable_denoise;
    float                              _bnr_table[XCAM_BNR_TABLE_SIZE];
    CLEeConfig                         _ee_config;
};

SmartPtr<CLImageHandler>
create_cl_bayer_pipe_image_handler (const SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_BAYER_PIPE_HANDLER_H
