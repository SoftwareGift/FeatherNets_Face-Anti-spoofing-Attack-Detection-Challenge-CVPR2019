/*
 * cl_3a_stats_calculator.cpp - CL 3a calculator
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

#include "xcam_utils.h"
#include "cl_3a_stats_calculator.h"

namespace XCam {

CL3AStatsCalculatorKernel::CL3AStatsCalculatorKernel (
    SmartPtr<CLContext> &context,
    SmartPtr<CL3AStatsCalculator> &image
)
    : CLImageKernel (context, "kernel_3a_stats")
    , _data_allocated (false)
    , _image (image)
{
    xcam_mem_clear (_stats_info);
}

XCamReturn
CL3AStatsCalculatorKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count,
    CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();
    const VideoBufferInfo & video_info = input->get_video_info ();

    XCAM_UNUSED (output);

    if (!_data_allocated && !allocate_data (video_info)) {
        XCAM_LOG_WARNING ("CL3AStatsCalculatorKernel allocate data failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    _output_buffer = output;
    _image_in = new CLVaImage (context, input);

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_stats_cl_buffer->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    arg_count = 2;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    // grid_size default 16
    work_size.global[0] = _stats_info.aligned_width;
    work_size.global[1] = _stats_info.aligned_height;
    work_size.local[0] = 8;
    work_size.local[1] = 1;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CL3AStatsCalculatorKernel::post_execute ()
{
    SmartPtr<CLContext> context = get_context ();
    SmartPtr<BufferProxy> buffer;
    SmartPtr<X3aStats> stats;
    SmartPtr<CLEvent>  event = new CLEvent;
    XCam3AStats *stats_ptr = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    context->finish ();
    _image_in.release ();
    //copy out and post 3a stats
    buffer = _stats_pool->get_buffer (_stats_pool);
    XCAM_FAIL_RETURN (WARNING, buffer.ptr (), XCAM_RETURN_ERROR_MEM, "3a stats pool stopped.");

    stats = buffer.dynamic_cast_ptr<X3aStats> ();
    XCAM_ASSERT (stats.ptr ());
    stats_ptr = stats->get_stats ();
    ret = _stats_cl_buffer->enqueue_read (
              stats_ptr->stats,
              0, _stats_info.aligned_width * _stats_info.aligned_height * sizeof (stats_ptr->stats[0]),
              CLEvent::EmptyList, event);

    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, ret, "3a stats enqueue read buffer failed.");
    XCAM_ASSERT (event->get_event_id ());

    ret = event->wait ();
    XCAM_FAIL_RETURN (WARNING, ret == XCAM_RETURN_NO_ERROR, ret, "3a stats buffer enqueue event wait failed");
    event.release ();

    stats->set_timestamp (_output_buffer->get_timestamp ());
    _output_buffer->attach_buffer (stats);
    //post stats out
    return _image->post_stats (stats);
}

void
CL3AStatsCalculatorKernel::pre_stop ()
{
    if (_stats_pool.ptr ())
        _stats_pool->stop ();
}

bool
CL3AStatsCalculatorKernel::allocate_data (const VideoBufferInfo &buffer_info)
{
    SmartPtr<CLContext> context = get_context ();

    _stats_pool = new X3aStatsPool ();
    _stats_pool->set_video_info (buffer_info);

    XCAM_FAIL_RETURN (
        WARNING,
        _stats_pool->reserve (32), // need reserve more if as attachement
        false,
        "reserve cl stats buffer failed");

    _stats_info = _stats_pool->get_stats_info ();
    _stats_cl_buffer = new CLBuffer (
        context,
        _stats_info.aligned_width * _stats_info.aligned_height * sizeof (XCamGridStat));

    XCAM_FAIL_RETURN (
        WARNING,
        _stats_cl_buffer->is_valid (),
        false,
        "allocate cl stats buffer failed");
    _data_allocated = true;

    return true;
}

CL3AStatsCalculator::CL3AStatsCalculator ()
    : CLImageHandler ("CL3AStatsCalculator")
{
}

XCamReturn
CL3AStatsCalculator::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    output = input;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CL3AStatsCalculator::post_stats (const SmartPtr<X3aStats> &stats)
{
    if (_stats_callback.ptr ())
        return _stats_callback->x3a_stats_ready (stats);

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImageHandler>
create_cl_3a_stats_image_handler (SmartPtr<CLContext> &context)
{
    SmartPtr<CL3AStatsCalculator> x3a_stats_handler;
    SmartPtr<CLImageKernel> x3a_stats_kernel;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    x3a_stats_handler = new CL3AStatsCalculator ();
    x3a_stats_kernel = new CL3AStatsCalculatorKernel (context, x3a_stats_handler);
    {
        XCAM_CL_KERNEL_FUNC_SOURCE_BEGIN(kernel_3a_stats)
#include "kernel_3a_stats.clx"
        XCAM_CL_KERNEL_FUNC_END;
        ret = x3a_stats_kernel->load_from_source (kernel_3a_stats_body, strlen (kernel_3a_stats_body));
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            NULL,
            "CL image handler(%s) load source failed", x3a_stats_kernel->get_kernel_name());
    }
    XCAM_ASSERT (x3a_stats_kernel->is_valid ());
    x3a_stats_handler->add_kernel (x3a_stats_kernel);

    return x3a_stats_handler;
}


};
