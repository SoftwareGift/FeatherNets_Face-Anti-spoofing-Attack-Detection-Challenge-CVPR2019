/*
 * cl_bayer_pipe_handler.cpp - CL bayer pipe handler
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
#include "xcam_utils.h"
#include "cl_bayer_pipe_handler.h"

#define SHARED_PIXEL_WIDTH 16
#define SHARED_PIXEL_HEIGHT 16

#define WORK_ITEM_X_SIZE 2
#define WORK_ITEM_Y_SIZE 2

namespace XCam {

CL3AStatsCalculatorContext::CL3AStatsCalculatorContext (const SmartPtr<CLContext> &context)
    : _context (context)
    , _stats_buf_index (0)
    , _data_allocated (false)
{
}

CL3AStatsCalculatorContext::~CL3AStatsCalculatorContext ()
{
}

bool
CL3AStatsCalculatorContext::allocate_data (const VideoBufferInfo &buffer_info)
{
    _stats_pool = new X3aStatsPool ();
    _stats_pool->set_video_info (buffer_info);

    XCAM_FAIL_RETURN (
        WARNING,
        _stats_pool->reserve (32), // need reserve more if as attachement
        false,
        "reserve cl stats buffer failed");

    _stats_info = _stats_pool->get_stats_info ();

    for (uint32_t i = 0; i < XCAM_CL_3A_STATS_BUFFER_COUNT; ++i) {
        _stats_cl_buffer[i] = new CLBuffer (
            _context,
            _stats_info.aligned_width * _stats_info.aligned_height * sizeof (XCamGridStat));

        XCAM_ASSERT (_stats_cl_buffer[i].ptr ());
        XCAM_FAIL_RETURN (
            WARNING,
            _stats_cl_buffer[i]->is_valid (),
            false,
            "allocate cl stats buffer failed");
    }
    _stats_buf_index = 0;
    _data_allocated = true;

    return true;
}

void
CL3AStatsCalculatorContext::pre_stop ()
{
    if (_stats_pool.ptr ())
        _stats_pool->stop ();
}


void
CL3AStatsCalculatorContext::clean_up_data ()
{
    _data_allocated = false;

    for (uint32_t i = 0; i < XCAM_CL_3A_STATS_BUFFER_COUNT; ++i) {
        _stats_cl_buffer[i].release ();
    }
    _stats_buf_index = 0;
}

SmartPtr<CLBuffer>
CL3AStatsCalculatorContext::get_next_buffer ()
{
    SmartPtr<CLBuffer> buf = _stats_cl_buffer[_stats_buf_index];
    _stats_buf_index = ((_stats_buf_index + 1) % XCAM_CL_3A_STATS_BUFFER_COUNT);
    return buf;
}

void debug_print_3a_stats (XCam3AStats *stats_ptr)
{

    for (int y = 30; y < 40; ++y) {
        printf ("---- y ");
        for (int x = 54; x < 64; ++x)
            printf ("%3d", stats_ptr->stats[y * stats_ptr->info.aligned_width + x].avg_y);
        printf ("\n");
    }

#if 0
#define DUMP_STATS(ch, w, h, aligned_w, stats) do {                 \
        printf ("stats " #ch ":");                                  \
        for (uint32_t y = 0; y < h; ++y) {                          \
            for (uint32_t x = 0; x < w; ++x)                        \
                printf ("%3d ", stats[y * aligned_w + x].avg_##ch); \
        }                                                           \
        printf ("\n");                           \
    } while (0)
    DUMP_STATS (r,  stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (gr, stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (gb, stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (b,  stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
    DUMP_STATS (y,  stats_ptr->info.width, stats_ptr->info.height,
                stats_ptr->info.aligned_width, stats_ptr->stats);
#endif
}

void debug_print_histogram (XCam3AStats *stats_ptr)
{
#define DUMP_HISTOGRAM(ch, bins, hist) do {      \
        printf ("histogram " #ch ":");           \
        for (uint32_t i = 0; i < bins; i++) {    \
            if (i % 16 == 0) printf ("\n");      \
            printf ("%4d ", hist[i].ch);         \
        }                                        \
        printf ("\n");                           \
    } while (0)

    DUMP_HISTOGRAM (r,  stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);
    DUMP_HISTOGRAM (gr, stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);
    DUMP_HISTOGRAM (gb, stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);
    DUMP_HISTOGRAM (b,  stats_ptr->info.histogram_bins, stats_ptr->hist_rgb);

    printf ("histogram y:");
    for (uint32_t i = 0; i < stats_ptr->info.histogram_bins; i++) {
        if (i % 16 == 0) printf ("\n");
        printf ("%4d ", stats_ptr->hist_y[i]);
    }
    printf ("\n");
}

SmartPtr<X3aStats>
CL3AStatsCalculatorContext::copy_stats_out (const SmartPtr<CLBuffer> &stats_cl_buf)
{
    SmartPtr<BufferProxy> buffer;
    SmartPtr<X3aStats> stats;
    SmartPtr<CLEvent>  event = new CLEvent;
    XCam3AStats *stats_ptr = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (stats_cl_buf.ptr ());

    buffer = _stats_pool->get_buffer (_stats_pool);
    XCAM_FAIL_RETURN (WARNING, buffer.ptr (), NULL, "3a stats pool stopped.");

    stats = buffer.dynamic_cast_ptr<X3aStats> ();
    XCAM_ASSERT (stats.ptr ());
    stats_ptr = stats->get_stats ();
    ret = stats_cl_buf->enqueue_read (
              stats_ptr->stats,
              0, _stats_info.aligned_width * _stats_info.aligned_height * sizeof (stats_ptr->stats[0]),
              CLEvent::EmptyList, event);

    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats enqueue read buffer failed.");
    XCAM_ASSERT (event->get_event_id ());

    ret = event->wait ();
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats buffer enqueue event wait failed");
    event.release ();

    //debug_print_3a_stats (stats_ptr);
    fill_histogram (stats_ptr);
    //debug_print_histogram (stats_ptr);

    return stats;
}

bool
CL3AStatsCalculatorContext::fill_histogram (XCam3AStats * stats)
{
    const XCam3AStatsInfo &stats_info = stats->info;
    const XCamGridStat *grid_stat;
    XCamHistogram *hist_rgb = stats->hist_rgb;
    uint32_t *hist_y = stats->hist_y;

    memset (hist_rgb, 0, sizeof(XCamHistogram) * stats_info.histogram_bins);
    memset (hist_y, 0, sizeof(uint32_t) * stats_info.histogram_bins);
    for (uint32_t i = 0; i < stats_info.width; i++) {
        for (uint32_t j = 0; j < stats_info.height; j++) {
            grid_stat = &stats->stats[j * stats_info.aligned_width + i];
            hist_rgb[grid_stat->avg_r].r++;
            hist_rgb[grid_stat->avg_gr].gr++;
            hist_rgb[grid_stat->avg_gb].gb++;
            hist_rgb[grid_stat->avg_b].b++;
            hist_y[grid_stat->avg_y]++;
        }
    }
    return true;
}

CLBayerPipeImageKernel::CLBayerPipeImageKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CLBayerPipeImageHandler> &handler)
    : CLImageKernel (context, "kernel_bayer_pipe")
    , _handler (handler)
{
    _blc_config.level_gr = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_r = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_b = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.level_gb = XCAM_CL_BLC_DEFAULT_LEVEL;
    _blc_config.color_bits = 8;

    _wb_config.r_gain = 1.0;
    _wb_config.gr_gain = 1.0;
    _wb_config.gb_gain = 1.0;
    _wb_config.b_gain = 1.0;

    for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; i++)
        _gamma_table[i] = (float)i / 256.0f;
    _gamma_table[XCAM_GAMMA_TABLE_SIZE] = 0.9999f;

    _3a_stats_context = new CL3AStatsCalculatorContext (context);
    XCAM_ASSERT (_3a_stats_context.ptr ());
}

bool
CLBayerPipeImageKernel::set_blc (const XCam3aResultBlackLevel &blc)
{
    _blc_config.level_r = (float)blc.r_level;
    _blc_config.level_gr = (float)blc.gr_level;
    _blc_config.level_gb = (float)blc.gb_level;
    _blc_config.level_b = (float)blc.b_level;
    //_blc_config.color_bits = 0;
    return true;
}

bool
CLBayerPipeImageKernel::set_wb (const XCam3aResultWhiteBalance &wb)
{
    _wb_config.r_gain = (float)wb.r_gain;
    _wb_config.gr_gain = (float)wb.gr_gain;
    _wb_config.gb_gain = (float)wb.gb_gain;
    _wb_config.b_gain = (float)wb.b_gain;
    return true;
}

bool
CLBayerPipeImageKernel::set_gamma_table (const XCam3aResultGammaTable &gamma)
{
    for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; i++)
        _gamma_table[i] = (float)gamma.table[i] / 256.0f;

    return true;
}

XCamReturn
CLBayerPipeImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & in_video_info = input->get_video_info ();
    const VideoBufferInfo & out_video_info = output->get_video_info ();

    if (!_3a_stats_context->is_ready () && !_3a_stats_context->allocate_data (in_video_info)) {
        XCAM_LOG_WARNING ("CL3AStatsCalculatorContext allocate data failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    _blc_config.color_bits = in_video_info.color_bits;

    _gamma_table_buffer = new CLBuffer(
        context, sizeof(float) * (XCAM_GAMMA_TABLE_SIZE + 1),
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR , &_gamma_table);

    _stats_cl_buffer = _3a_stats_context->get_next_buffer ();

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);

    args[2].arg_adress = &_blc_config;
    args[2].arg_size = sizeof (_blc_config);

    args[3].arg_adress = &_wb_config;
    args[3].arg_size = sizeof (_wb_config);

    args[4].arg_adress = &_gamma_table_buffer->get_mem_id ();
    args[4].arg_size = sizeof (cl_mem);

    args[5].arg_adress = &_stats_cl_buffer->get_mem_id ();
    args[5].arg_size = sizeof (cl_mem);
    arg_count = 6;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = XCAM_ALIGN_UP(out_video_info.width, 16) / WORK_ITEM_X_SIZE;
    work_size.global[1] = XCAM_ALIGN_UP(out_video_info.height, 16) / WORK_ITEM_Y_SIZE;
    work_size.local[0] = SHARED_PIXEL_WIDTH / WORK_ITEM_X_SIZE;
    work_size.local[1] = SHARED_PIXEL_HEIGHT / WORK_ITEM_Y_SIZE;

    _output_buffer = output;

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLBayerPipeImageKernel::post_execute ()
{
    SmartPtr<X3aStats> stats_3a;

    _image_in.release ();
    _image_out.release ();
    _gamma_table_buffer.release ();


    stats_3a = _3a_stats_context->copy_stats_out (_stats_cl_buffer);
    if (!stats_3a.ptr ()) {
        XCAM_LOG_DEBUG ("copy 3a stats failed, maybe handler stopped");
        return XCAM_RETURN_ERROR_CL;
    }

    stats_3a->set_timestamp (_output_buffer->get_timestamp ());
    _output_buffer->attach_buffer (stats_3a);

    _stats_cl_buffer.release ();
    _output_buffer.release ();

    XCAM_FAIL_RETURN (WARNING, stats_3a.ptr (), XCAM_RETURN_ERROR_MEM, "3a stats dequeue failed");
    //return XCAM_RETURN_NO_ERROR;
    return _handler->post_stats (stats_3a);
}

void
CLBayerPipeImageKernel::pre_stop ()
{
    _3a_stats_context->pre_stop ();
}

CLBayerPipeImageHandler::CLBayerPipeImageHandler (const char *name)
    : CLImageHandler (name)
    , _output_format (XCAM_PIX_FMT_RGBA64)
{
}

bool
CLBayerPipeImageHandler::set_output_format (uint32_t fourcc)
{
    XCAM_FAIL_RETURN (
        WARNING,
        fourcc == XCAM_PIX_FMT_RGBA64 || fourcc == V4L2_PIX_FMT_RGB24 ||
        fourcc == V4L2_PIX_FMT_XBGR32 || fourcc == V4L2_PIX_FMT_ABGR32 || V4L2_PIX_FMT_BGR32 ||
        //fourcc == V4L2_PIX_FMT_RGB32 || fourcc == V4L2_PIX_FMT_ARGB32 || V4L2_PIX_FMT_XRGB32 ||
        fourcc == V4L2_PIX_FMT_RGBA32,
        false,
        "CL image handler(%s) doesn't support format(%s) settings",
        get_name (), xcam_fourcc_to_string (fourcc));

    _output_format = fourcc;
    return true;
}

bool
CLBayerPipeImageHandler::set_bayer_kernel (SmartPtr<CLBayerPipeImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _bayer_kernel = kernel;
    return true;
}

bool
CLBayerPipeImageHandler::set_blc_config (const XCam3aResultBlackLevel &blc)
{
    return _bayer_kernel->set_blc (blc);
}

bool
CLBayerPipeImageHandler::set_wb_config (const XCam3aResultWhiteBalance &wb)
{
    return _bayer_kernel->set_wb (wb);
}

bool
CLBayerPipeImageHandler::set_gamma_table (const XCam3aResultGammaTable &gamma)
{
    return _bayer_kernel->set_gamma_table (gamma);
}

XCamReturn
CLBayerPipeImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    uint32_t format = _output_format;
    bool format_inited = output.init (format, input.width, input.height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) ouput format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (format));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLBayerPipeImageHandler::post_stats (const SmartPtr<X3aStats> &stats)
{
    if (_stats_callback.ptr ())
        return _stats_callback->x3a_stats_ready (stats);

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_bayer_pipe_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CLBayerPipeImageHandler> bayer_pipe_handler;
    SmartPtr<CLBayerPipeImageKernel> bayer_pipe_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    bayer_pipe_handler = new CLBayerPipeImageHandler ("cl_handler_bayer_pipe");
    bayer_pipe_kernel = new CLBayerPipeImageKernel (context, bayer_pipe_handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_bayer_pipe)
#include "kernel_bayer_pipe.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = bayer_pipe_kernel->load_from_source (kernel_bayer_pipe_body, strlen (kernel_bayer_pipe_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", bayer_pipe_kernel->get_kernel_name());
    }
    XCAM_ASSERT (bayer_pipe_kernel->is_valid ());
    bayer_pipe_handler->set_bayer_kernel (bayer_pipe_kernel);

    return bayer_pipe_handler;
}

};
