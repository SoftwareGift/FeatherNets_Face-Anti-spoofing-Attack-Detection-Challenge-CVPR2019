/*
 * cl_image_handler.cpp - CL image handler
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

#include "cl_image_handler.h"
#if HAVE_LIBDRM
#include "drm_display.h"
#include "cl_image_bo_buffer.h"
#include "drm_bo_buffer.h"
#endif
#include "cl_device.h"
#include "swapped_buffer.h"

namespace XCam {

#define XCAM_CL_IMAGE_HANDLER_DEFAULT_BUF_NUM 4

CLImageKernel::CLImageKernel (const SmartPtr<CLContext> &context, const char *name, bool enable)
    : CLKernel (context, name)
    , _enable (enable)
{
}

CLImageKernel::~CLImageKernel ()
{
}

/*
 * Default kernel arguments
 * arg0:
 *     input,   __read_only image2d_t
 * arg1:
 *     output,  __write_only image2d_t
 * suppose cl can get width/height pixels from
 * get_image_width/get_image_height
 */
XCamReturn
CLImageKernel::pre_execute ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    CLArgList args;
    CLWorkSize work_size;

    XCAM_FAIL_RETURN (
        ERROR, !is_arguments_set (), XCAM_RETURN_ERROR_PARAM,
        "cl image kernel(%s) pre_execute failed since arguments was set somewhere", get_kernel_name ());

    ret = prepare_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR, ret,
        "cl image kernel(%s) prepare arguments failed", get_kernel_name ());

    ret = set_arguments (args, work_size);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR, ret,
        "cl image kernel(%s) set_arguments failed", get_kernel_name ());

    return ret;
}

XCamReturn
CLImageKernel::prepare_arguments (
    CLArgList &args, CLWorkSize &work_size)
{
    XCAM_UNUSED (args);
    XCAM_UNUSED (work_size);

    XCAM_LOG_ERROR (
        "cl image kernel(%s) prepare_arguments error."
        "Did you forget to set_arguments or prepare_arguments was not derived", get_kernel_name ());
    return XCAM_RETURN_ERROR_CL;
}

CLImageHandler::CLImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : _name (NULL)
    , _enable (true)
    , _context (context)
    , _buf_pool_type (CLImageHandler::CLVideoPoolType)
    , _disable_buf_pool (false)
    , _buf_pool_size (XCAM_CL_IMAGE_HANDLER_DEFAULT_BUF_NUM)
    , _buf_swap_flags ((uint32_t)(SwappedBuffer::OrderY0Y1) | (uint32_t)(SwappedBuffer::OrderUV0UV1))
    , _buf_swap_init_order (SwappedBuffer::OrderY0Y1)
    , _result_timestamp (XCam::InvalidTimestamp)
{
    XCAM_ASSERT (name);
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    XCAM_OBJ_PROFILING_INIT;
}

CLImageHandler::~CLImageHandler ()
{
    if (_name)
        xcam_free (_name);
}

bool
CLImageHandler::enable_buf_pool_swap_flags (
    uint32_t flags,
    uint32_t init_order)
{
#if HAVE_LIBDRM
    _buf_swap_flags = flags;
    _buf_swap_init_order = init_order;

    SmartPtr<DrmBoBufferPool> pool = _buf_pool.dynamic_cast_ptr<DrmBoBufferPool> ();

    if (pool.ptr () && !pool->update_swap_init_order (init_order)) {
        XCAM_LOG_ERROR (
            "Handler(%s) update swap order(0x%04x) to buffer pool failed",
            XCAM_STR (get_name ()),
            init_order);
        return false;
    }
    return true;
#else
    XCAM_LOG_ERROR ("CLImageHandler doesn't support swapping flags");

    XCAM_UNUSED (flags);
    XCAM_UNUSED (init_order);
    return false;
#endif
}

bool
CLImageHandler::add_kernel (const SmartPtr<CLImageKernel> &kernel)
{
    _kernels.push_back (kernel);
    return true;
}

bool
CLImageHandler::enable_handler (bool enable)
{
    _enable = enable;
    return true;
}

bool
CLImageHandler::is_handler_enabled () const
{
    return _enable;
}

XCamReturn
CLImageHandler::create_buffer_pool (const VideoBufferInfo &video_info)
{
    if (_buf_pool.ptr ())
        return XCAM_RETURN_ERROR_PARAM;

    if (_buf_pool_type == CLImageHandler::CLVideoPoolType) {
        SmartPtr<BufferPool> pool = new CLVideoBufferPool ();
        _buf_pool = pool.ptr() ? pool : _buf_pool;
    }
#if HAVE_LIBDRM
    else {
        SmartPtr<DrmDisplay> display = DrmDisplay::instance ();
        XCAM_FAIL_RETURN(
            WARNING,
            display.ptr (),
            XCAM_RETURN_ERROR_CL,
            "CLImageHandler(%s) failed to get drm dispay", XCAM_STR (_name));

        if (_buf_pool_type == CLImageHandler::DrmBoPoolType) {
            SmartPtr<BufferPool> pool = new DrmBoBufferPool (display);
            _buf_pool = pool.ptr() ? pool : _buf_pool;
        } else if (_buf_pool_type == CLImageHandler::CLBoPoolType) {
            SmartPtr<BufferPool> pool = new CLBoBufferPool (display, get_context ());
            _buf_pool = pool.ptr() ? pool : _buf_pool;
        }
    }
#endif
    XCAM_FAIL_RETURN(
        WARNING,
        _buf_pool.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLImageHandler(%s) create buffer pool failed, pool_type:%d",
        XCAM_STR (_name), (int32_t)_buf_pool_type);

    // buffer_pool->set_swap_flags (_buf_swap_flags, _buf_swap_init_order);
    _buf_pool->set_video_info (video_info);
    XCAM_FAIL_RETURN(
        WARNING,
        _buf_pool->reserve (_buf_pool_size),
        XCAM_RETURN_ERROR_CL,
        "CLImageHandler(%s) failed to init drm buffer pool", XCAM_STR (_name));

    return XCAM_RETURN_NO_ERROR;
}

bool CLImageHandler::is_ready ()
{
    if (_disable_buf_pool)
        return true;
    if (!_buf_pool.ptr ())  //execute not triggered
        return true;
    if (_buf_pool->has_free_buffers ())
        return true;
    return false;
}

XCamReturn CLImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    output = input;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageHandler::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    XCAM_ASSERT (input.ptr () && output.ptr ());
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageHandler::ensure_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = prepare_parameters (input, output);
    XCAM_FAIL_RETURN(
        WARNING, ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS, ret,
        "CLImageHandler(%s) failed to prepare_parameters", XCAM_STR (_name));

    reset_buf_cache (input, output);
    return ret;
}

void
CLImageHandler::reset_buf_cache (const SmartPtr<VideoBuffer>& input, const SmartPtr<VideoBuffer>& output)
{
    _input_buf_cache = input;
    _output_buf_cache = output;
}

XCamReturn
CLImageHandler::prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (_disable_buf_pool)
        return XCAM_RETURN_NO_ERROR;

    if (!_buf_pool.ptr ()) {
        VideoBufferInfo output_video_info;

        ret = prepare_buffer_pool_video_info (input->get_video_info (), output_video_info);
        XCAM_FAIL_RETURN(
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            ret,
            "CLImageHandler(%s) prepare output video info failed", XCAM_STR (_name));

        ret = create_buffer_pool (output_video_info);
        XCAM_FAIL_RETURN(
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            ret,
            "CLImageHandler(%s) ensure drm buffer pool failed", XCAM_STR (_name));
    }

    output = _buf_pool->get_buffer (_buf_pool);
    XCAM_FAIL_RETURN(
        WARNING,
        output.ptr(),
        XCAM_RETURN_ERROR_UNKNOWN,
        "CLImageHandler(%s) failed to get drm buffer from pool", XCAM_STR (_name));

    // TODO, need consider output is not sync up with input buffer
    output->set_timestamp (input->get_timestamp ());
    output->copy_attaches (input);

    return XCAM_RETURN_NO_ERROR;
}

void
CLImageHandler::emit_stop ()
{
    for (KernelList::iterator i_kernel = _kernels.begin ();
            i_kernel != _kernels.end ();  ++i_kernel) {
        (*i_kernel)->pre_stop ();
    }

    if (_buf_pool.ptr ())
        _buf_pool->stop ();
}

SmartPtr<VideoBuffer> &
CLImageHandler::get_input_buf ()
{
    XCAM_ASSERT (_input_buf_cache.ptr ());
    return _input_buf_cache;
}

SmartPtr<VideoBuffer> &
CLImageHandler::get_output_buf ()
{
    XCAM_ASSERT (_output_buf_cache.ptr ());
    return _output_buf_cache;
}

XCamReturn
CLImageHandler::execute_kernel (SmartPtr<CLImageKernel> &kernel)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (!kernel->is_enabled ())
        return XCAM_RETURN_NO_ERROR;

    if (!kernel->is_arguments_set ()) {
        XCAM_FAIL_RETURN (
            WARNING,
            (ret = kernel->pre_execute ()) == XCAM_RETURN_NO_ERROR, ret,
            "cl_image_handler(%s) pre_execute kernel(%s) failed",
            XCAM_STR (_name), kernel->get_kernel_name ());
    }

    CLArgList args = kernel->get_args ();
    ret = kernel->execute (kernel, false);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS, ret,
        "cl_image_handler(%s) execute kernel(%s) failed",
        XCAM_STR (_name), kernel->get_kernel_name ());

#if 0
    ret = kernel->post_execute (args);
    XCAM_FAIL_RETURN (
        WARNING,
        (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS),
        ret,
        "cl_image_handler(%s) post_execute kernel(%s) failed",
        XCAM_STR (_name), kernel->get_kernel_name ());
#endif

    return ret;
}

XCamReturn
CLImageHandler::execute_kernels ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    for (KernelList::iterator i_kernel = _kernels.begin ();
            i_kernel != _kernels.end (); ++i_kernel) {
        SmartPtr<CLImageKernel> &kernel = *i_kernel;

        XCAM_FAIL_RETURN (
            WARNING, kernel.ptr(), XCAM_RETURN_ERROR_PARAM,
            "kernel empty");

        ret = execute_kernel (kernel);

        if (ret != XCAM_RETURN_NO_ERROR)
            break;
    }

    return ret;
}

XCamReturn
CLImageHandler::execute (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        WARNING,
        !_kernels.empty (),
        XCAM_RETURN_ERROR_PARAM,
        "cl_image_handler(%s) no image kernel set", XCAM_STR (_name));

    if (!is_handler_enabled ()) {
        output = input;
        return XCAM_RETURN_NO_ERROR;
    }

    XCAM_FAIL_RETURN (
        WARNING,
        (ret = prepare_output_buf (input, output)) == XCAM_RETURN_NO_ERROR,
        ret,
        "cl_image_handler (%s) prepare output buf failed", XCAM_STR (_name));
    XCAM_ASSERT (output.ptr ());

    ret = ensure_parameters (input, output);
    XCAM_FAIL_RETURN (
        WARNING, (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS), ret,
        "cl_image_handler (%s) ensure parameters failed", XCAM_STR (_name));

    if (ret == XCAM_RETURN_BYPASS)
        return ret;

    XCAM_OBJ_PROFILING_START;
    ret = execute_kernels ();

    reset_buf_cache (NULL, NULL);

#if ENABLE_PROFILING
    get_context ()->finish ();
#endif
    XCAM_OBJ_PROFILING_END (XCAM_STR (_name), XCAM_OBJ_DUR_FRAME_NUM);

    XCAM_FAIL_RETURN (
        WARNING, (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS), ret,
        "cl_image_handler (%s) execute kernels failed", XCAM_STR (_name));

    if (ret != XCAM_RETURN_NO_ERROR)
        return ret;

    ret = execute_done (output);
    return ret;
}

XCamReturn
CLImageHandler::execute_done (SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (output);
    return XCAM_RETURN_NO_ERROR;
}

void
CLImageHandler::set_3a_result (SmartPtr<X3aResult> &result)
{
    if (!result.ptr ())
        return;

    int64_t ts = result->get_timestamp ();
    _result_timestamp = (ts != XCam::InvalidTimestamp) ? ts : _result_timestamp;

    X3aResultList::iterator i_res = _3a_results.begin ();
    for (; i_res != _3a_results.end(); ++i_res) {
        if (result->get_type () == (*i_res)->get_type ()) {
            (*i_res) = result;
            break;
        }
    }

    if (i_res == _3a_results.end ()) {
        _3a_results.push_back (result);
    }
}

SmartPtr<X3aResult>
CLImageHandler::get_3a_result (XCam3aResultType type)
{
    X3aResultList::iterator i_res = _3a_results.begin ();
    SmartPtr<X3aResult> res;

    for ( ; i_res != _3a_results.end(); ++i_res) {
        if (type == (*i_res)->get_type ()) {
            res = (*i_res);
            break;
        }
    }
    return res;
}

bool
CLImageHandler::append_kernels (SmartPtr<CLImageHandler> handler)
{
    XCAM_ASSERT (!handler->_kernels.empty ());
    _kernels.insert (_kernels.end (), handler->_kernels.begin (), handler->_kernels.end ());
    return true;
}

CLCloneImageHandler::CLCloneImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
    , _clone_flags (SwappedBuffer::SwapNone)
{
}

XCamReturn
CLCloneImageHandler::prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
#if HAVE_LIBDRM
    XCAM_FAIL_RETURN (
        ERROR,
        _clone_flags != (uint32_t)(SwappedBuffer::SwapNone),
        XCAM_RETURN_ERROR_PARAM,
        "CLCloneImageHandler(%s) clone output buffer failed since clone_flags none",
        XCAM_STR (get_name ()));

    XCAM_ASSERT (input.ptr ());
    SmartPtr<SwappedBuffer> swap_input = input.dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_ASSERT (swap_input.ptr ());
    SmartPtr<SwappedBuffer> swap_output = swap_input->swap_clone (swap_input, _clone_flags);
    SmartPtr<DrmBoBuffer> swapped_buf = swap_output.dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_FAIL_RETURN (
        ERROR,
        swapped_buf.ptr (),
        XCAM_RETURN_ERROR_UNKNOWN,
        "CLCloneImageHandler(%s) clone output buffer failed(clone_flags:%d)",
        XCAM_STR (get_name ()), _clone_flags);

    output = swapped_buf;
    return XCAM_RETURN_NO_ERROR;
#else
    XCAM_LOG_ERROR ("CLCloneImageHandler doesn't support DrmBoBuffer");

    XCAM_UNUSED (input);
    XCAM_UNUSED (output);
    return XCAM_RETURN_ERROR_PARAM;
#endif
}

};
