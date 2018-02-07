/*
 * cl_bayer_basic_handler.cpp - CL bayer basic handler
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

#include "cl_utils.h"
#include "cl_bayer_basic_handler.h"
#include "xcam_thread.h"

#define GROUP_CELL_X_SIZE 64
#define GROUP_CELL_Y_SIZE 4

#define STATS_3A_CELL_X_SIZE 8
#define STATS_3A_CELL_Y_SIZE GROUP_CELL_Y_SIZE

#define STANDARD_3A_STATS_SIZE 8

#define ENABLE_IMAGE_2D_INPUT 0

namespace XCam {

static const XCamKernelInfo kernel_bayer_basic_info = {
    "kernel_bayer_basic",
#include "kernel_bayer_basic.clx"
    , 0,
};

struct BayerPostData {
    SmartPtr<VideoBuffer> image_buffer;
    SmartPtr<CLBuffer>    stats_cl_buf;
};

class CLBayer3AStatsThread
    : public Thread
{
public:
    CLBayer3AStatsThread (CLBayerBasicImageHandler *handler)
        : Thread ("CLBayer3AStatsThread")
        , _handler (handler)
    {}
    ~CLBayer3AStatsThread () {}

    virtual bool emit_stop ();
    bool queue_stats (SmartPtr<VideoBuffer> &buf, SmartPtr<CLBuffer> &stats);
    SmartPtr<VideoBuffer> pop_buf ();
protected:
    virtual bool loop ();
    virtual void stopped ();

private:
    CLBayerBasicImageHandler     *_handler;
    SafeList<BayerPostData>      _stats_process_list;
    SafeList<VideoBuffer>        _buffer_done_list;
};

bool
CLBayer3AStatsThread::emit_stop ()
{
    _stats_process_list.pause_pop ();
    _buffer_done_list.pause_pop ();

    _stats_process_list.wakeup ();
    _buffer_done_list.wakeup ();

    return Thread::emit_stop ();
}

bool
CLBayer3AStatsThread::queue_stats (SmartPtr<VideoBuffer> &buf, SmartPtr<CLBuffer> &stats)
{
    XCAM_FAIL_RETURN (
        WARNING,
        buf.ptr () && stats.ptr (),
        false,
        "cl bayer 3a-stats thread has error buffer/stats to queue");

    SmartPtr<BayerPostData> data = new BayerPostData;
    XCAM_ASSERT (data.ptr ());
    data->image_buffer = buf;
    data->stats_cl_buf = stats;

    return _stats_process_list.push (data);
}

SmartPtr<VideoBuffer>
CLBayer3AStatsThread::pop_buf ()
{
    return _buffer_done_list.pop ();
}

void
CLBayer3AStatsThread::stopped ()
{
    _stats_process_list.clear ();
    _buffer_done_list.clear ();
}

bool
CLBayer3AStatsThread::loop ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<BayerPostData> data;
    data = _stats_process_list.pop ();
    if (!data.ptr ()) {
        XCAM_LOG_INFO ("cl bayer 3a-stats thread is going to stop, processing data empty");
        return false;
    }

    XCAM_ASSERT (data->image_buffer.ptr ());
    XCAM_ASSERT (data->stats_cl_buf.ptr ());
    XCAM_ASSERT (_handler);

    ret = _handler->process_stats_buffer (data->image_buffer, data->stats_cl_buf);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        false,
        "cl bayer 3a-stats thread has error buffer on kernel post processing");

    XCAM_FAIL_RETURN (
        ERROR,
        _buffer_done_list.push (data->image_buffer),
        false,
        "cl bayer 3a-stats thread failed to queue done-buffers");
    return true;
}

CLBayerBasicImageKernel::CLBayerBasicImageKernel (const SmartPtr<CLContext> &context)
    : CLImageKernel (context, "kernel_bayer_basic")
{
}

XCamReturn
CLBayerBasicImageHandler::process_stats_buffer (SmartPtr<VideoBuffer> &buffer, SmartPtr<CLBuffer> &cl_stats)
{
    SmartPtr<X3aStats> stats_3a;
    SmartPtr<CLContext> context = get_context ();

    XCAM_OBJ_PROFILING_START;

    context->finish ();
    stats_3a = _3a_stats_context->copy_stats_out (cl_stats);
    if (!stats_3a.ptr ()) {
        XCAM_LOG_DEBUG ("copy 3a stats failed, maybe handler stopped");
        return XCAM_RETURN_ERROR_CL;
    }

    stats_3a->set_timestamp (buffer->get_timestamp ());
    buffer->attach_buffer (stats_3a);

    if (cl_stats.ptr ())
        _3a_stats_context->release_buffer (cl_stats);

    XCAM_OBJ_PROFILING_END ("3a_stats_cpu_copy(async)", XCAM_OBJ_DUR_FRAME_NUM);

    return post_stats (stats_3a);
}

CLBayerBasicImageHandler::CLBayerBasicImageHandler (
    const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _is_first_buf (true)
{
    _blc_config.level_gr = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_r = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_b = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_gb = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.color_bits = 10;

    _wb_config.r_gain = 1.0;
    _wb_config.gr_gain = 1.0;
    _wb_config.gb_gain = 1.0;
    _wb_config.b_gain = 1.0;

    for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; i++)
        _gamma_table[i] = (float)i / 256.0f;
    _gamma_table[XCAM_GAMMA_TABLE_SIZE] = 0.9999f;

    SmartPtr<CL3AStatsCalculatorContext> stats_context = new CL3AStatsCalculatorContext (context);
    XCAM_ASSERT (stats_context.ptr ());
    _3a_stats_context = stats_context;

    SmartPtr<CLBayer3AStatsThread> stats_thread = new CLBayer3AStatsThread (this);
    XCAM_ASSERT (stats_thread.ptr ());
    _3a_stats_thread = stats_thread;

    XCAM_OBJ_PROFILING_INIT;
}

CLBayerBasicImageHandler::~CLBayerBasicImageHandler ()
{
    _3a_stats_thread->stop ();
    _3a_stats_context->clean_up_data ();
}

void
CLBayerBasicImageHandler::set_stats_bits (uint32_t stats_bits)
{
    XCAM_ASSERT (_3a_stats_context.ptr ());
    _3a_stats_context->set_bit_depth (stats_bits);
}

bool
CLBayerBasicImageHandler::set_bayer_kernel (SmartPtr<CLBayerBasicImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _bayer_kernel = kernel;
    return true;
}

bool
CLBayerBasicImageHandler::set_blc_config (const XCam3aResultBlackLevel &blc)
{
    _blc_config.level_r = (float)blc.r_level;
    _blc_config.level_gr = (float)blc.gr_level;
    _blc_config.level_gb = (float)blc.gb_level;
    _blc_config.level_b = (float)blc.b_level;
    //_blc_config.color_bits = 0;
    return true;
}

bool
CLBayerBasicImageHandler::set_wb_config (const XCam3aResultWhiteBalance &wb)
{
    _wb_config.r_gain = (float)wb.r_gain;
    _wb_config.gr_gain = (float)wb.gr_gain;
    _wb_config.gb_gain = (float)wb.gb_gain;
    _wb_config.b_gain = (float)wb.b_gain;
    return true;
}

bool
CLBayerBasicImageHandler::set_gamma_table (const XCam3aResultGammaTable &gamma)
{
    for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; i++)
        _gamma_table[i] = (float)gamma.table[i] / 256.0f;

    return true;
}

void
CLBayerBasicImageHandler::emit_stop ()
{
    _3a_stats_context->pre_stop ();
    _3a_stats_thread->emit_stop ();
}

XCamReturn
CLBayerBasicImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t format = XCAM_PIX_FMT_SGRBG16_planar;
    bool format_inited = output.init (format, input.width / 2 , input.height / 2);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) output format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (format));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLBayerBasicImageHandler::prepare_parameters (
    SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();
    CLImageDesc in_image_info;
    CLImageDesc out_image_info;
    CLArgList args;
    CLWorkSize work_size;

    XCAM_ASSERT (_bayer_kernel.ptr ());

    if (!_3a_stats_context->is_ready () &&
            !_3a_stats_context->allocate_data (
                in_video_info,
                STANDARD_3A_STATS_SIZE / STATS_3A_CELL_X_SIZE,
                STANDARD_3A_STATS_SIZE / STATS_3A_CELL_Y_SIZE)) {
        XCAM_LOG_WARNING ("CL3AStatsCalculatorContext allocate data failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    if (_is_first_buf) {
        XCAM_FAIL_RETURN (
            WARNING, _3a_stats_thread->start (), XCAM_RETURN_ERROR_THREAD,
            "cl bayer basic handler start 3a stats thread failed");
    }

    in_image_info.format.image_channel_order = CL_RGBA;
    in_image_info.format.image_channel_data_type = CL_UNSIGNED_INT32; //CL_UNORM_INT16;
    in_image_info.width = in_video_info.aligned_width / 8;
    in_image_info.height = in_video_info.height;
    in_image_info.row_pitch = in_video_info.strides[0];

    out_image_info.format.image_channel_order = CL_RGBA;
    out_image_info.format.image_channel_data_type = CL_UNSIGNED_INT32; //CL_UNORM_INT16;
    out_image_info.width = out_video_info.width  / 8;
    out_image_info.height = out_video_info.aligned_height * 4;
    out_image_info.row_pitch = out_video_info.strides[0];

#if ENABLE_IMAGE_2D_INPUT
    SmartPtr<CLImage> image_in = convert_to_climage (context, input, in_image_info);
#else
    SmartPtr<CLBuffer> buffer_in = convert_to_clbuffer (context, input);
#endif
    uint32_t input_aligned_width = in_video_info.strides[0] / (2 * 8); // ushort8
    SmartPtr<CLImage> image_out = convert_to_climage (context, output, out_image_info);

    uint32_t out_aligned_height = out_video_info.aligned_height;
    _blc_config.color_bits = in_video_info.color_bits;

    SmartPtr<CLBuffer> gamma_table_buffer = new CLBuffer(
        context, sizeof(float) * (XCAM_GAMMA_TABLE_SIZE + 1),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_gamma_table);

    _stats_cl_buffer = _3a_stats_context->get_buffer ();
    XCAM_FAIL_RETURN (
        WARNING,
        _stats_cl_buffer.ptr () && _stats_cl_buffer->is_valid (),
        XCAM_RETURN_ERROR_PARAM,
        "CLBayerBasic handler get 3a stats buffer failed");

    XCAM_FAIL_RETURN (
        WARNING,
        image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image handler(%s) out memory not available", XCAM_STR(get_name ()));

    //set args;
#if ENABLE_IMAGE_2D_INPUT
    args.push_back (new CLMemArgument (image_in));
#else
    args.push_back (new CLMemArgument (buffer_in));
#endif
    args.push_back (new CLArgumentT<uint32_t> (input_aligned_width));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLArgumentT<uint32_t> (out_aligned_height));
    args.push_back (new CLArgumentT<CLBLCConfig> (_blc_config));
    args.push_back (new CLArgumentT<CLWBConfig> (_wb_config));
    args.push_back (new CLMemArgument (gamma_table_buffer));
    args.push_back (new CLMemArgument (_stats_cl_buffer));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 2;
    work_size.global[0] = XCAM_ALIGN_UP(out_video_info.width, GROUP_CELL_X_SIZE) / GROUP_CELL_X_SIZE * work_size.local[0];
    work_size.global[1] = XCAM_ALIGN_UP(out_video_info.aligned_height, GROUP_CELL_Y_SIZE) / GROUP_CELL_Y_SIZE * work_size.local[1];

    //printf ("work_size:g(%d, %d), l(%d, %d)\n", work_size.global[0], work_size.global[1], work_size.local[0], work_size.local[1]);
    XCAM_ASSERT (_bayer_kernel.ptr ());
    XCamReturn ret = _bayer_kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "bayer basic kernel set arguments failed.");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLBayerBasicImageHandler::execute_done (SmartPtr<VideoBuffer> &output)
{
    XCAM_FAIL_RETURN (
        ERROR, _3a_stats_thread->queue_stats (output, _stats_cl_buffer), XCAM_RETURN_ERROR_UNKNOWN,
        "cl bayer basic handler(%s) process 3a stats failed", XCAM_STR (get_name ()));

    _stats_cl_buffer.release ();

    if (_is_first_buf) {
        _is_first_buf = false;
        return XCAM_RETURN_BYPASS;
    }

    SmartPtr<VideoBuffer> done_buf = _3a_stats_thread->pop_buf ();
    XCAM_FAIL_RETURN (
        WARNING,
        done_buf.ptr (),
        XCAM_RETURN_ERROR_MEM,
        "cl bayer handler(%s) failed to get done buffer", get_name ());
    output = done_buf;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLBayerBasicImageHandler::post_stats (const SmartPtr<X3aStats> &stats)
{
    if (_stats_callback.ptr ())
        return _stats_callback->x3a_stats_ready (stats);

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_bayer_basic_image_handler (const SmartPtr<CLContext> &context, bool enable_gamma, uint32_t stats_bits)
{
    SmartPtr<CLBayerBasicImageHandler> bayer_planar_handler;
    SmartPtr<CLBayerBasicImageKernel> basic_kernel;
    char build_options[1024];

    bayer_planar_handler = new CLBayerBasicImageHandler (context, "cl_handler_bayer_basic");
    bayer_planar_handler->set_stats_bits (stats_bits);
    basic_kernel = new CLBayerBasicImageKernel (context);
    XCAM_ASSERT (basic_kernel.ptr ());

    xcam_mem_clear (build_options);
    snprintf (build_options, sizeof (build_options),
              " -DENABLE_GAMMA=%d "
              " -DENABLE_IMAGE_2D_INPUT=%d "
              " -DSTATS_BITS=%d ",
              (enable_gamma ? 1 : 0),
              ENABLE_IMAGE_2D_INPUT,
              stats_bits);
    XCAM_FAIL_RETURN (
        ERROR, basic_kernel->build_kernel (kernel_bayer_basic_info, build_options) == XCAM_RETURN_NO_ERROR, NULL,
        "build bayer-basic kernel(%s) failed", kernel_bayer_basic_info.kernel_name);

    XCAM_ASSERT (basic_kernel->is_valid ());
    bayer_planar_handler->set_bayer_kernel (basic_kernel);

    return bayer_planar_handler;
}

};
