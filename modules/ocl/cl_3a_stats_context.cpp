/*
 * cl_3a_stats_context.cpp - CL 3a stats context
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
 * Author: Jia Meng <jia.meng@intel.com>
 */

#include <xcam_std.h>
#include "cl_3a_stats_context.h"

namespace XCam {
CL3AStatsCalculatorContext::CL3AStatsCalculatorContext (const SmartPtr<CLContext> &context)
    : _context (context)
    , _width_factor (1)
    , _height_factor (1)
    , _factor_shift (0)
    , _data_allocated (false)
{
    SmartPtr<X3aStatsPool> pool = new X3aStatsPool ();
    XCAM_ASSERT (pool.ptr ());
    _stats_pool = pool;
}

CL3AStatsCalculatorContext::~CL3AStatsCalculatorContext ()
{
    clean_up_data ();
}

void
CL3AStatsCalculatorContext::set_bit_depth (uint32_t bits)
{
    XCAM_ASSERT (_stats_pool.ptr ());
    _stats_pool->set_bit_depth (bits);
}

bool
CL3AStatsCalculatorContext::allocate_data (const VideoBufferInfo &buffer_info, uint32_t width_factor, uint32_t height_factor)
{
    uint32_t multiply_factor = 0;

    _stats_pool->set_video_info (buffer_info);

    XCAM_FAIL_RETURN (
        WARNING,
        _stats_pool->reserve (32), // need reserve more if as attachement
        false,
        "reserve cl stats buffer failed");

    _stats_info = _stats_pool->get_stats_info ();
    XCAM_ASSERT ((width_factor & (width_factor - 1)) == 0 &&
                 (height_factor & (height_factor - 1)) == 0);
    _width_factor = width_factor;
    _height_factor = height_factor;
    multiply_factor = width_factor * height_factor;
    _factor_shift = 0;
    while ((multiply_factor >>= 1) != 0) {
        ++_factor_shift;
    }

    _stats_mem_size =
        _stats_info.aligned_width * _width_factor *
        _stats_info.aligned_height * _height_factor * sizeof (CL3AStatsStruct);

    for (uint32_t i = 0; i < XCAM_CL_3A_STATS_BUFFER_COUNT; ++i) {
        SmartPtr<CLBuffer> buf_new = new CLBuffer (
            _context, _stats_mem_size);

        XCAM_ASSERT (buf_new.ptr ());
        XCAM_FAIL_RETURN (
            WARNING,
            buf_new->is_valid (),
            false,
            "allocate cl stats buffer failed");
        _stats_cl_buffers.push (buf_new);
    }
    _data_allocated = true;

    return true;
}

void
CL3AStatsCalculatorContext::pre_stop ()
{
    if (_stats_pool.ptr ())
        _stats_pool->stop ();
    _stats_cl_buffers.pause_pop ();
    _stats_cl_buffers.wakeup ();
}

void
CL3AStatsCalculatorContext::clean_up_data ()
{
    _data_allocated = false;

    _stats_cl_buffers.pause_pop ();
    _stats_cl_buffers.wakeup ();
    _stats_cl_buffers.clear ();
}

SmartPtr<CLBuffer>
CL3AStatsCalculatorContext::get_buffer ()
{
    SmartPtr<CLBuffer> buf = _stats_cl_buffers.pop ();
    return buf;
}

bool
CL3AStatsCalculatorContext::release_buffer (SmartPtr<CLBuffer> &buf)
{
    XCAM_ASSERT (buf.ptr ());
    if (!buf.ptr ())
        return false;
    return _stats_cl_buffers.push (buf);
}

void debug_print_3a_stats (XCam3AStats *stats_ptr)
{
    static int frames = 0;
    frames++;
    printf ("********frame(%d) debug 3a stats(%dbits) \n", frames, stats_ptr->info.bit_depth);
    for (int y = 30; y < 60; ++y) {
        printf ("---- y ");
        for (int x = 40; x < 80; ++x)
            printf ("%4d ", stats_ptr->stats[y * stats_ptr->info.aligned_width + x].avg_y);
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
    SmartPtr<VideoBuffer> buffer;
    SmartPtr<X3aStats> stats;
    SmartPtr<CLEvent>  event = new CLEvent;
    XCam3AStats *stats_ptr = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    void *buf_ptr = NULL;
    const CL3AStatsStruct *cl_buf_ptr = NULL;

    XCAM_ASSERT (stats_cl_buf.ptr ());

    buffer = _stats_pool->get_buffer (_stats_pool);
    XCAM_FAIL_RETURN (WARNING, buffer.ptr (), NULL, "3a stats pool stopped.");

    stats = buffer.dynamic_cast_ptr<X3aStats> ();
    XCAM_ASSERT (stats.ptr ());
    stats_ptr = stats->get_stats ();

    ret = stats_cl_buf->enqueue_map (
              buf_ptr,
              0, _stats_mem_size,
              CL_MAP_READ,
              CLEvent::EmptyList,
              event);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats enqueue read buffer failed.");
    XCAM_ASSERT (event->get_event_id ());
    ret = event->wait ();
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats buffer enqueue event wait failed");

    cl_buf_ptr = (const CL3AStatsStruct*)buf_ptr;

    XCAM_ASSERT (stats_ptr);
    memset (stats_ptr->stats, 0, sizeof (XCamGridStat) * _stats_info.aligned_width * _stats_info.aligned_height);
    //uint32_t avg_factor = _width_factor * _height_factor;
    //uint32_t avg_round_pading = avg_factor / 2;
    uint32_t cl_stats_width = _stats_info.aligned_width * _width_factor;

    for (uint32_t h = 0; h < _stats_info.height; ++h) {
        XCamGridStat *grid_stats_line = &stats_ptr->stats[_stats_info.aligned_width * h];
        uint32_t end_i_h = (h + 1) * _height_factor;
        for (uint32_t i_h = h * _height_factor; i_h < end_i_h; ++i_h) {
            const CL3AStatsStruct *cl_stats_line = &cl_buf_ptr[cl_stats_width * i_h];
            for (uint32_t w = 0; w < _stats_info.width; ++w) {
                uint32_t end_i_w = (w + 1) * _width_factor;
                for (uint32_t i_w = w * _width_factor; i_w < end_i_w; ++i_w) {
                    //grid_stats_line[w].avg_y += (cl_stats_line[i_w].avg_y + avg_round_pading) / avg_factor;
                    grid_stats_line[w].avg_y += (cl_stats_line[i_w].avg_y >> _factor_shift);
                    grid_stats_line[w].avg_r += (cl_stats_line[i_w].avg_r >> _factor_shift);
                    grid_stats_line[w].avg_gr += (cl_stats_line[i_w].avg_gr >> _factor_shift);
                    grid_stats_line[w].avg_gb += (cl_stats_line[i_w].avg_gb >> _factor_shift);
                    grid_stats_line[w].avg_b += (cl_stats_line[i_w].avg_b >> _factor_shift);
                    grid_stats_line[w].valid_wb_count += cl_stats_line[i_w].valid_wb_count;
                    grid_stats_line[w].f_value1 += cl_stats_line[i_w].f_value1;
                    grid_stats_line[w].f_value2 += cl_stats_line[i_w].f_value2;
                }
            }
        }
    }

    event.release ();

    SmartPtr<CLEvent>  unmap_event = new CLEvent;
    ret = stats_cl_buf->enqueue_unmap (buf_ptr, CLEvent::EmptyList, unmap_event);
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats buffer enqueue unmap failed");
    ret = unmap_event->wait ();
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, NULL, "3a stats buffer enqueue unmap event wait failed");
    unmap_event.release ();
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

}
