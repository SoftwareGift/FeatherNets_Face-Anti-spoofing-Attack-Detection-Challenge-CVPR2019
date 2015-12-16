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

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "cl_blc_handler.h"
#include "cl_wb_handler.h"
#include "cl_ee_handler.h"
#include "stats_callback_interface.h"
#include "x3a_stats_pool.h"
#include "cl_context.h"
#include "cl_3a_stats_calculator.h"

#define XCAM_BNR_TABLE_SIZE 64

namespace XCam {

#if 0
#define XCAM_CL_BLC_DEFAULT_LEVEL 0.06

/*  Black level correction configuration  */
typedef struct  {
    float     level_gr;  /* Black level for GR pixels */
    float     level_r;   /* Black level for R pixels */
    float     level_b;   /* Black level for B pixels */
    float     level_gb;  /* Black level for GB pixels */
    uint32_t  color_bits;
} CLBLCConfig;

typedef struct {
    float           r_gain;
    float           gr_gain;
    float           gb_gain;
    float           b_gain;
} CLWBConfig;
#endif

class CLBayerPipeImageHandler;

class CLBayerPipeImageKernel
    : public CLImageKernel
{
public:
    explicit CLBayerPipeImageKernel (
        SmartPtr<CLContext> &context,
        SmartPtr<CLBayerPipeImageHandler> &handler);

    bool enable_denoise (bool enable);
    bool set_ee (const XCam3aResultEdgeEnhancement &ee);
    bool set_bnr (const XCam3aResultBayerNoiseReduction &bnr);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLBayerPipeImageKernel);

private:
    uint32_t                  _input_height;
    uint32_t                  _output_height;
    uint32_t                  _enable_denoise;
    float                     _bnr_table[XCAM_BNR_TABLE_SIZE];
    SmartPtr<CLBuffer>        _bnr_table_buffer;
    CLEeConfig                _ee_config;

    SmartPtr<CLBayerPipeImageHandler>     _handler;
};

class CLBayerPipeImageHandler
    : public CLImageHandler
{
    friend class CLBayerPipeImageKernel;

public:
    explicit CLBayerPipeImageHandler (const char *name);
    bool set_bayer_kernel (SmartPtr<CLBayerPipeImageKernel> &kernel);
    bool set_ee_config (const XCam3aResultEdgeEnhancement &ee);
    bool set_bnr_config (const XCam3aResultBayerNoiseReduction &bnr);
    ;
    bool set_output_format (uint32_t fourcc);
    bool enable_denoise (bool enable);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCAM_DEAD_COPY (CLBayerPipeImageHandler);

private:
    SmartPtr<CLBayerPipeImageKernel>   _bayer_kernel;
    uint32_t                           _output_format;
};

SmartPtr<CLImageHandler>
create_cl_bayer_pipe_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_BAYER_PIPE_HANDLER_H
