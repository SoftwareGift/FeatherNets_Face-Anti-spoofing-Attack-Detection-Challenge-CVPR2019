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
#include "stats_callback_interface.h"
#include "x3a_stats_pool.h"
#include "cl_context.h"
#include "cl_3a_stats_calculator.h"

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

class CL3AStatsCalculatorContext
{
public:
    CL3AStatsCalculatorContext (const SmartPtr<CLContext> &context);
    ~CL3AStatsCalculatorContext ();

    bool is_ready () const {
        return _data_allocated;
    }
    bool allocate_data (const VideoBufferInfo &buffer_info);
    void pre_stop ();
    void clean_up_data ();

    SmartPtr<CLBuffer> get_next_buffer ();
    SmartPtr<X3aStats> copy_stats_out (const SmartPtr<CLBuffer> &stats_cl_buf);

private:
    XCAM_DEAD_COPY (CL3AStatsCalculatorContext);

    bool fill_histogram (XCam3AStats *stats);

private:
    SmartPtr<CLContext>              _context;
    SmartPtr<X3aStatsPool>           _stats_pool;
    SmartPtr<CLBuffer>               _stats_cl_buffer[XCAM_CL_3A_STATS_BUFFER_COUNT];
    uint32_t                         _stats_buf_index;
    XCam3AStatsInfo                  _stats_info;
    bool                             _data_allocated;
};

class CLBayerPipeImageKernel
    : public CLImageKernel
{
public:
    explicit CLBayerPipeImageKernel (
        SmartPtr<CLContext> &context,
        SmartPtr<CLBayerPipeImageHandler> &handler);

    bool set_blc (const XCam3aResultBlackLevel &blc);
    bool set_wb (const XCam3aResultWhiteBalance &wb);
    bool set_gamma_table (const XCam3aResultGammaTable &gamma);
    bool enable_denoise (bool enable);
    bool enable_gamma (bool enable);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

    virtual XCamReturn post_execute ();
    virtual void pre_stop ();

private:
    XCAM_DEAD_COPY (CLBayerPipeImageKernel);

private:
    uint32_t                  _output_height;
    CLBLCConfig               _blc_config;
    CLWBConfig                _wb_config;
    uint32_t                  _enable_denoise;
    uint32_t                  _enable_gamma;
    float                     _gamma_table[XCAM_GAMMA_TABLE_SIZE + 1];
    SmartPtr<CLBuffer>        _gamma_table_buffer;
    SmartPtr<CL3AStatsCalculatorContext>  _3a_stats_context;
    SmartPtr<CLBuffer>        _stats_cl_buffer;

    SmartPtr<DrmBoBuffer>     _output_buffer;

    SmartPtr<CLBayerPipeImageHandler>     _handler;
};

class CLBayerPipeImageHandler
    : public CLImageHandler
{
    friend class CLBayerPipeImageKernel;

public:
    explicit CLBayerPipeImageHandler (const char *name);
    void set_stats_callback (SmartPtr<StatsCallback> &callback) {
        _stats_callback = callback;
    }
    bool set_bayer_kernel (SmartPtr<CLBayerPipeImageKernel> &kernel);

    bool set_output_format (uint32_t fourcc);
    bool set_blc_config (const XCam3aResultBlackLevel &blc);
    bool set_wb_config (const XCam3aResultWhiteBalance &wb);
    bool set_gamma_table (const XCam3aResultGammaTable &gamma);
    bool enable_denoise (bool enable);
    bool enable_gamma (bool enable);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCamReturn post_stats (const SmartPtr<X3aStats> &stats);

    XCAM_DEAD_COPY (CLBayerPipeImageHandler);

private:
    SmartPtr<CLBayerPipeImageKernel>   _bayer_kernel;
    uint32_t                           _output_format;
    SmartPtr<StatsCallback>            _stats_callback;
};

SmartPtr<CLImageHandler>
create_cl_bayer_pipe_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_BAYER_PIPE_HANDLER_H
