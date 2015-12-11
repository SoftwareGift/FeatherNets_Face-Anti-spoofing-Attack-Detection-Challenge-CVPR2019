/*
 * cl_bayer_basic_handler.h - CL bayer copy handler
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
 */

#ifndef XCAM_CL_BAYER_BASIC_HANLDER_H
#define XCAM_CL_BAYER_BASIC_HANLDER_H

#include "xcam_utils.h"
#include "cl_blc_handler.h"
#include "cl_wb_handler.h"
#include "cl_image_handler.h"
#include "cl_memory.h"
#include "cl_3a_stats_context.h"
#include "stats_callback_interface.h"

namespace XCam {

class CLBayerBasicImageHandler;

class CLBayerBasicImageKernel
    : public CLImageKernel
{
public:
    explicit CLBayerBasicImageKernel (SmartPtr<CLContext> &context, SmartPtr<CLBayerBasicImageHandler>& handler);
    void set_stats_bits (uint32_t stats_bits);

    bool set_blc (const XCam3aResultBlackLevel &blc);
    bool set_wb (const XCam3aResultWhiteBalance &wb);
    bool set_gamma_table (const XCam3aResultGammaTable &gamma);

    virtual XCamReturn post_execute ();
    virtual void pre_stop ();

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLBayerBasicImageKernel);

private:
    uint32_t                  _input_aligned_width;
    uint32_t                  _out_aligned_height;
    SmartPtr<CLBuffer>        _buffer_in;
    CLBLCConfig               _blc_config;
    CLWBConfig                _wb_config;

    float                     _gamma_table[XCAM_GAMMA_TABLE_SIZE + 1];
    SmartPtr<CLBuffer>        _gamma_table_buffer;
    SmartPtr<DrmBoBuffer>     _output_buffer;

    SmartPtr<CLBuffer>        _stats_cl_buffer;
    SmartPtr<CL3AStatsCalculatorContext>  _3a_stats_context;
    SmartPtr<CLBayerBasicImageHandler>    _handler;

};

class CLBayerBasicImageHandler
    : public CLImageHandler
{

public:
    explicit CLBayerBasicImageHandler (const char *name);

    void set_stats_callback (SmartPtr<StatsCallback> &callback) {
        _stats_callback = callback;
    }
    bool set_bayer_kernel (SmartPtr<CLBayerBasicImageKernel> &kernel);

    bool set_blc_config (const XCam3aResultBlackLevel &blc);
    bool set_wb_config (const XCam3aResultWhiteBalance &wb);
    bool set_gamma_table (const XCam3aResultGammaTable &gamma);

    XCamReturn post_stats (const SmartPtr<X3aStats> &stats);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    SmartPtr<CLBayerBasicImageKernel>   _bayer_kernel;

    SmartPtr<StatsCallback>            _stats_callback;
};


SmartPtr<CLImageHandler>
create_cl_bayer_basic_image_handler (
    SmartPtr<CLContext> &context,
    bool enable_gamma = true,
    uint32_t stats_bits = 8);

};

#endif //XCAM_CL_BAYER_BASIC_HANLDER_H