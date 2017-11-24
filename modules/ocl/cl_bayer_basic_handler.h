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

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_memory.h>
#include <ocl/cl_3a_stats_context.h>
#include <stats_callback_interface.h>

namespace XCam {

class CLBayerBasicImageHandler;
class CLBayer3AStatsThread;

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

class CLBayerBasicImageKernel
    : public CLImageKernel
{
public:
    explicit CLBayerBasicImageKernel (const SmartPtr<CLContext> &context);
};

class CLBayerBasicImageHandler
    : public CLImageHandler
{
    friend class CLBayer3AStatsThread;
public:
    explicit CLBayerBasicImageHandler (const SmartPtr<CLContext> &context, const char *name);
    ~CLBayerBasicImageHandler ();

    void set_stats_callback (SmartPtr<StatsCallback> &callback) {
        _stats_callback = callback;
    }
    bool set_bayer_kernel (SmartPtr<CLBayerBasicImageKernel> &kernel);

    bool set_blc_config (const XCam3aResultBlackLevel &blc);
    bool set_wb_config (const XCam3aResultWhiteBalance &wb);
    bool set_gamma_table (const XCam3aResultGammaTable &gamma);
    void set_stats_bits (uint32_t stats_bits);

    virtual void emit_stop ();
    XCamReturn post_stats (const SmartPtr<X3aStats> &stats);
    XCamReturn process_stats_buffer (SmartPtr<VideoBuffer> &buffer, SmartPtr<CLBuffer> &cl_stats);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input, VideoBufferInfo &output);
    virtual XCamReturn prepare_parameters (
        SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

private:
    SmartPtr<CLBayerBasicImageKernel>     _bayer_kernel;
    bool                                  _is_first_buf;
    CLBLCConfig                           _blc_config;
    CLWBConfig                            _wb_config;
    float                                 _gamma_table[XCAM_GAMMA_TABLE_SIZE + 1];

    SmartPtr<CL3AStatsCalculatorContext>  _3a_stats_context;
    SmartPtr<CLBayer3AStatsThread>        _3a_stats_thread;
    SmartPtr<CLBuffer>                    _stats_cl_buffer;

    SmartPtr<StatsCallback>               _stats_callback;

    XCAM_OBJ_PROFILING_DEFINES;
};

SmartPtr<CLImageHandler>
create_cl_bayer_basic_image_handler (
    const SmartPtr<CLContext> &context,
    bool enable_gamma = true,
    uint32_t stats_bits = 8);

};

#endif //XCAM_CL_BAYER_BASIC_HANLDER_H
