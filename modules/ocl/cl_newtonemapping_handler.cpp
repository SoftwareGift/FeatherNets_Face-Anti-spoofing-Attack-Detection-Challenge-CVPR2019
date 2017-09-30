/*
 * cl_newtonemapping_handler.cpp - CL tonemapping handler
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
 *  Author: Wu Junkai <junkai.wu@intel.com>
 */

#include "cl_utils.h"
#include "cl_newtonemapping_handler.h"

namespace XCam {

static const XCamKernelInfo kernel_tone_mapping_pipe_info = {
    "kernel_newtonemapping",
#include "kernel_newtonemapping.clx"
    , 0,
};

CLNewTonemappingImageKernel::CLNewTonemappingImageKernel (
    const SmartPtr<CLContext> &context, const char *name)
    : CLImageKernel (context, name)
{
}

static void
haleq(int *y, int *hist, int *hist_leq, int left, int right, int level, int index_left, int index_right)
{
    int l;
    float e, le;

    l = (left + right) / 2;
    int num_left = left > 0 ? hist[left - 1] : 0;
    int pixel_num = hist[right] - num_left;
    e = y[num_left + pixel_num / 2];

    if(e != 0)
    {
        le = 0.5f * (e - l) + l;
    }
    else
    {
        le = l;
    }

    int index = (index_left + index_right) / 2;
    hist_leq[index] = (int)(le + 0.5f);

    if(level > 5) return;

    haleq (y, hist, hist_leq, left, (int)(le + 0.5f), level + 1, index_left, index);
    haleq (y, hist, hist_leq, (int)(le + 0.5f) + 1, right, level + 1, index + 1, index_right);
}

static void
block_split_haleq(int* hist, int hist_bin_count, int pixel_num, int block_start_index, float* y_max, float* y_avg, float* map_hist)
{
    int block_id = block_start_index / hist_bin_count;

    for(int i = hist_bin_count - 1; i >= 0; i--)
    {
        if(hist[i] > 0)
        {
            y_max[block_id] = i;
            break;
        }
    }

    for(int i = 0; i < hist_bin_count; i++)
    {
        y_avg[block_id] += i * hist[i];
    }

    y_max[block_id] = y_max[block_id] + 1;
    y_avg[block_id] = y_avg[block_id] / pixel_num;

    int *hist_log = (int *) xcam_malloc0 (hist_bin_count * sizeof (int));
    int *sort_y = (int *) xcam_malloc0 ((pixel_num + 1) * sizeof (int));
    int *map_index_leq = (int *) xcam_malloc0 (hist_bin_count * sizeof (int));
    int *map_index_log = (int *) xcam_malloc0 (hist_bin_count * sizeof (int));
    XCAM_ASSERT (hist_log && sort_y && map_index_leq && map_index_log);

    int thres = (int)(1500 * 1500 / (y_avg[block_id] * y_avg[block_id] + 1) * 600);
    int y_max0 = (y_max[block_id] > thres) ? thres : y_max[block_id];
    int y_max1 = (y_max[block_id] - thres) > 0 ? (y_max[block_id] - thres) : 0;

    float t0 = 0.01f * y_max0 + 0.001f;
    float t1 = 0.001f * y_max1 + 0.001f;
    float max0_log = log(y_max0 + t0);
    float max1_log = log(y_max1 + t1);
    float t0_log = log(t0);
    float t1_log = log(t1);
    float factor0;

    if(y_max[block_id] < thres)
    {
        factor0 = (hist_bin_count - 1) / (max0_log - t0_log + 0.001f);
    }
    else
        factor0 = y_max0 / (max0_log - t0_log + 0.001f);

    float factor1 = y_max1 / (max1_log - t1_log + 0.001f);

    if(y_max[block_id] < thres)
    {
        for(int i = 0; i < y_max[block_id]; i++)
        {
            int index = (int)((log(i + t0) - t0_log) * factor0 + 0.5f);
            hist_log[index] += hist[i];
            map_index_log[i] = index;
        }
    }
    else
    {
        for(int i = 0; i < y_max0; i++)
        {
            int index = (int)((log(i + t0) - t0_log) * factor0 + 0.5f);
            hist_log[index] += hist[i];
            map_index_log[i] = index;
        }

        for(int i = y_max0; i < y_max[block_id]; i++)
        {
            int r = y_max[block_id] - i;
            int index = (int)((log(r + t1) - t1_log) * factor1 + 0.5f);
            index = y_max[block_id] - index;
            hist_log[index] += hist[i];
            map_index_log[i] = index;
        }
    }

    for(int i = y_max[block_id]; i < hist_bin_count; i++)
    {
        hist_log[map_index_log[(int)y_max[block_id] - 1]] += hist[i];
        map_index_log[i] = map_index_log[(int)y_max[block_id] - 1];
    }

    int sort_index = 1;
    for(int i = 0; i < hist_bin_count; i++)
    {
        for(int l = 0; l < hist_log[i]; l++)
        {
            sort_y[sort_index] = i;
            sort_index++;
        }
    }
    sort_y[0] = 0;

    for(int i = 1; i < hist_bin_count; i++)
    {
        hist_log[i] += hist_log[i - 1];
    }

    int map_leq_index[256];

    haleq(sort_y, hist_log, map_leq_index, 0, hist_bin_count - 1, 0, 0, 255);

    map_leq_index[255] = hist_bin_count;
    map_leq_index[0] = 0;

    for(int i = 1; i < 255; i++)
    {
        if(i % 2 == 0) map_leq_index[i] = (map_leq_index[i - 1] + map_leq_index[i + 1]) / 2;
        if(map_leq_index[i] < map_leq_index[i - 1])
            map_leq_index[i] = map_leq_index[i - 1];
    }

    for(int i = 0; i < 255; i++)
    {
        for(int k = map_leq_index[i]; k < map_leq_index[i + 1]; k++)
        {
            map_index_leq[k] = (float)i;
        }
    }

    for(int i = 0; i < hist_bin_count; i++)
    {
        map_hist[i + block_start_index] = map_index_leq[map_index_log[i]] / 255.0f;
    }

    y_max[block_id] = y_max[block_id] / hist_bin_count;
    y_avg[block_id] = y_avg[block_id] / hist_bin_count;

    xcam_free (hist_log);
    hist_log = NULL;
    xcam_free (map_index_leq);
    map_index_leq = NULL;
    xcam_free (map_index_log);
    map_index_log = NULL;
    xcam_free (sort_y);
    sort_y = NULL;
}

CLNewTonemappingImageHandler::CLNewTonemappingImageHandler (
    const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _output_format (XCAM_PIX_FMT_SGRBG16_planar)
    , _block_factor (4)
{
    for(int i = 0; i < 65536; i++)
    {
        _map_hist[i] = i;
    }

    for(int i = 0; i < 4 * 4; i++)
    {
        _y_max[i] = 0.0f;
        _y_avg[i] = 0.0f;
    }
}

bool
CLNewTonemappingImageHandler::set_tonemapping_kernel(SmartPtr<CLNewTonemappingImageKernel> &kernel)
{
    SmartPtr<CLImageKernel> image_kernel = kernel;
    add_kernel (image_kernel);
    _tonemapping_kernel = kernel;
    return true;
}

XCamReturn
CLNewTonemappingImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    bool format_inited = output.init (_output_format, input.width, input.height);

    XCAM_FAIL_RETURN (
        WARNING,
        format_inited,
        XCAM_RETURN_ERROR_PARAM,
        "CL image handler(%s) output format(%s) unsupported",
        get_name (), xcam_fourcc_to_string (_output_format));

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLNewTonemappingImageHandler::prepare_parameters (
    SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo &video_info = input->get_video_info ();
    CLArgList args;
    CLWorkSize work_size;

    XCAM_ASSERT (_tonemapping_kernel.ptr ());

    CLImageDesc desc;
    desc.format.image_channel_order = CL_RGBA;
    desc.format.image_channel_data_type = CL_UNORM_INT16;
    desc.width = video_info.aligned_width / 4;
    desc.height = video_info.aligned_height * 4;
    desc.row_pitch = video_info.strides[0];
    desc.array_size = 4;
    desc.slice_pitch = video_info.strides [0] * video_info.aligned_height;

    SmartPtr<CLImage> image_in = convert_to_climage (context, input, desc);
    SmartPtr<CLImage> image_out = convert_to_climage (context, output, desc);
    int image_width = video_info.aligned_width;
    int image_height = video_info.aligned_height;

    XCAM_FAIL_RETURN (
        WARNING,
        image_in->is_valid () && image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image handler(%s) in/out memory not available", XCAM_STR (get_name ()));

    SmartPtr<X3aStats> stats;
    SmartPtr<CLVideoBuffer> cl_buf = input.dynamic_cast_ptr<CLVideoBuffer> ();
    if (cl_buf.ptr ()) {
        stats = cl_buf->find_3a_stats ();
    }
#if HAVE_LIBDRM
    else {
        SmartPtr<DrmBoBuffer> bo_buf = input.dynamic_cast_ptr<DrmBoBuffer> ();
        stats = bo_buf->find_3a_stats ();
    }
#endif
    XCAM_FAIL_RETURN (
        ERROR, stats.ptr (), XCAM_RETURN_ERROR_MEM,
        "new tonemapping handler prepare_arguments find_3a_stats failed");

    XCam3AStats *stats_ptr = stats->get_stats ();
    XCAM_FAIL_RETURN (
        ERROR, stats_ptr, XCAM_RETURN_ERROR_MEM,
        "new tonemapping handler prepare_arguments get_stats failed");

    int block_factor = 4;
    int width_per_block = stats_ptr->info.width / block_factor;
    int height_per_block = stats_ptr->info.height / block_factor;
    int height_last_block = height_per_block + stats_ptr->info.height % block_factor;
    int hist_bin_count = 1 << stats_ptr->info.bit_depth;

    int *hist_per_block = (int *) xcam_malloc0 (hist_bin_count * sizeof (int));
    XCAM_ASSERT (hist_per_block);

    for(int block_row = 0; block_row < block_factor; block_row++)
    {
        for(int block_col = 0; block_col < block_factor; block_col++)
        {
            int block_start_index = (block_row * block_factor + block_col) * hist_bin_count;
            int start_index = block_row * height_per_block * stats_ptr->info.width + block_col * width_per_block;

            for(int i = 0; i < hist_bin_count; i++)
            {
                hist_per_block[i] = 0;
            }

            if(block_row == block_factor - 1)
            {
                height_per_block = height_last_block;
            }

            int block_totalnum = width_per_block * height_per_block;
            for(int i = 0; i < height_per_block; i++)
            {
                for(int j = 0; j < width_per_block; j++)
                {
                    int y = stats_ptr->stats[start_index + i * stats_ptr->info.width + j].avg_y;
                    hist_per_block[y]++;
                }
            }

            block_split_haleq (hist_per_block, hist_bin_count, block_totalnum, block_start_index, _y_max, _y_avg, _map_hist);
        }
    }

    xcam_free (hist_per_block);
    hist_per_block = NULL;

    SmartPtr<CLBuffer> y_max_buffer = new CLBuffer(
        context, sizeof(float) * block_factor * block_factor,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_y_max);

    SmartPtr<CLBuffer> y_avg_buffer = new CLBuffer(
        context, sizeof(float) * block_factor * block_factor,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_y_avg);

    SmartPtr<CLBuffer> map_hist_buffer = new CLBuffer(
        context, sizeof(float) * hist_bin_count * block_factor * block_factor,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, &_map_hist);

    //set args;
    args.push_back (new CLMemArgument (image_in));
    args.push_back (new CLMemArgument (image_out));
    args.push_back (new CLMemArgument (y_max_buffer));
    args.push_back (new CLMemArgument (y_avg_buffer));
    args.push_back (new CLMemArgument (map_hist_buffer));
    args.push_back (new CLArgumentT<int> (image_width));
    args.push_back (new CLArgumentT<int> (image_height));

    const CLImageDesc out_info = image_out->get_image_desc ();
    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.global[0] = out_info.width;
    work_size.global[1] = out_info.height / 4;
    work_size.local[0] = 8;
    work_size.local[1] = 8;

    XCAM_ASSERT (_tonemapping_kernel.ptr ());
    XCamReturn ret = _tonemapping_kernel->set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "new tone mapping kernel set arguments failed.");

    return XCAM_RETURN_NO_ERROR;
}


SmartPtr<CLImageHandler>
create_cl_newtonemapping_image_handler (const SmartPtr<CLContext> &context)
{
    SmartPtr<CLNewTonemappingImageHandler> tonemapping_handler;
    SmartPtr<CLNewTonemappingImageKernel> tonemapping_kernel;

    tonemapping_kernel = new CLNewTonemappingImageKernel (context, "kernel_newtonemapping");
    XCAM_ASSERT (tonemapping_kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, tonemapping_kernel->build_kernel (kernel_tone_mapping_pipe_info, NULL) == XCAM_RETURN_NO_ERROR, NULL,
        "build new tonemapping kernel(%s) failed", kernel_tone_mapping_pipe_info.kernel_name);

    XCAM_ASSERT (tonemapping_kernel->is_valid ());
    tonemapping_handler = new CLNewTonemappingImageHandler(context, "cl_handler_newtonemapping");
    tonemapping_handler->set_tonemapping_kernel(tonemapping_kernel);

    return tonemapping_handler;
}

};
