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
#include "drm_display.h"
#include "cl_device.h"


namespace XCam {

#define XCAM_CL_IMAGE_HANDLER_DEFAULT_BUF_NUM 6

CLWorkSize::CLWorkSize ()
    : dim (XCAM_DEFAULT_IMAGE_DIM)
{
    xcam_mem_clear (global);
    xcam_mem_clear (local);
}

CLArgument::CLArgument()
    : arg_adress (NULL)
    , arg_size (0)
{
}

CLImageKernel::CLImageKernel (SmartPtr<CLContext> &context, const char *name, bool enable)
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
CLImageKernel::pre_execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLContext> context = get_context ();
#define XCAM_CL_MAX_ARGS 10
    CLArgument args[XCAM_CL_MAX_ARGS];
    uint32_t arg_count = XCAM_CL_MAX_ARGS;
    CLWorkSize work_size;

    ret = prepare_arguments (input, output, args, arg_count, work_size);

    XCAM_ASSERT (arg_count);
    for (uint32_t i = 0; i < arg_count; ++i) {
        ret = set_argument (i, args[i].arg_adress, args[i].arg_size);
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            ret,
            "cl image kernel(%s) set argc(%d) failed", get_kernel_name (), i);
    }

    XCAM_ASSERT (work_size.global[0]);
    ret = set_work_size (work_size.dim, work_size.global, work_size.local);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "cl image kernel(%s) set work size failed", get_kernel_name ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageKernel::prepare_arguments (
    SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
    CLArgument args[], uint32_t &arg_count, CLWorkSize &work_size)
{
    SmartPtr<CLContext> context = get_context ();

    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    args[0].arg_adress = &_image_in->get_mem_id ();
    args[0].arg_size = sizeof (cl_mem);
    args[1].arg_adress = &_image_out->get_mem_id ();
    args[1].arg_size = sizeof (cl_mem);
    arg_count = 2;

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    {
        const CLImageDesc &out_info = _image_out->get_image_desc ();
        work_size.global[0] = out_info.width;
        work_size.global[1] = out_info.height;
    }
    work_size.local[0] = 0;
    work_size.local[1] = 0;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageKernel::post_execute ()
{
    _image_in.release ();
    _image_out.release ();
    return XCAM_RETURN_NO_ERROR;
}

CLImageHandler::CLImageHandler (const char *name)
    : _name (NULL)
{
    XCAM_ASSERT (name);
    if (name)
        _name = strdup (name);

    XCAM_OBJ_PROFILING_INIT;
}

CLImageHandler::~CLImageHandler ()
{
    if (_name)
        xcam_free (_name);
}

bool
CLImageHandler::add_kernel (SmartPtr<CLImageKernel> &kernel)
{
    _kernels.push_back (kernel);
    return true;
}

bool
CLImageHandler::set_kernels_enable (bool enable)
{
    for (KernelList::iterator i_kernel = _kernels.begin ();
            i_kernel != _kernels.end (); ++i_kernel) {
        (*i_kernel)->set_enable (enable);
    }

    return true;
}

bool
CLImageHandler::is_kernels_enabled () const
{
    for (KernelList::const_iterator i_kernel = _kernels.begin ();
            i_kernel != _kernels.end (); ++i_kernel) {
        if ((*i_kernel)->is_enabled ())
            return true;
    }

    return false;
}

XCamReturn
CLImageHandler::create_buffer_pool (const VideoBufferInfo &video_info)
{
    SmartPtr<BufferPool> buffer_pool;
    SmartPtr<DrmDisplay> display;

    if (_buf_pool.ptr ())
        return XCAM_RETURN_ERROR_PARAM;

    display = DrmDisplay::instance ();
    XCAM_FAIL_RETURN(
        WARNING,
        display.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLImageHandler(%s) failed to get drm dispay", XCAM_STR (_name));

    buffer_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (buffer_pool.ptr ());
    buffer_pool->set_video_info (video_info);

    XCAM_FAIL_RETURN(
        WARNING,
        buffer_pool->reserve (XCAM_CL_IMAGE_HANDLER_DEFAULT_BUF_NUM),
        XCAM_RETURN_ERROR_CL,
        "CLImageHandler(%s) failed to init drm buffer pool", XCAM_STR (_name));

    _buf_pool = buffer_pool;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn CLImageHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    output = input;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageHandler::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    SmartPtr<BufferProxy> new_buf;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

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

    new_buf = _buf_pool->get_buffer (_buf_pool);
    XCAM_FAIL_RETURN(
        WARNING,
        new_buf.ptr(),
        XCAM_RETURN_ERROR_UNKNOWN,
        "CLImageHandler(%s) failed to get drm buffer from pool", XCAM_STR (_name));

    new_buf->set_timestamp (input->get_timestamp ());
    new_buf->copy_attaches (input);

    output = new_buf.dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_ASSERT (output.ptr ());
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

XCamReturn
CLImageHandler::execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (
        WARNING,
        !_kernels.empty (),
        XCAM_RETURN_ERROR_PARAM,
        "cl_image_handler(%s) no image kernel set", XCAM_STR (_name));

    if (!is_kernels_enabled ()) {
        output = input;
        return XCAM_RETURN_NO_ERROR;
    }

    XCAM_OBJ_PROFILING_START;

    XCAM_FAIL_RETURN (
        WARNING,
        (ret = prepare_output_buf (input, output)) == XCAM_RETURN_NO_ERROR,
        ret,
        "cl_image_handler (%s) prepare output buf failed", XCAM_STR (_name));

    XCAM_ASSERT (output.ptr ());

    for (KernelList::iterator i_kernel = _kernels.begin ();
            i_kernel != _kernels.end (); ++i_kernel) {
        SmartPtr<CLImageKernel> &kernel = *i_kernel;

        XCAM_FAIL_RETURN (
            WARNING,
            kernel.ptr(),
            ret,
            "kernel empty");

        if (!kernel->is_enabled ())
            continue;

        XCAM_FAIL_RETURN (
            WARNING,
            (ret = kernel->pre_execute (input, output)) == XCAM_RETURN_NO_ERROR,
            ret,
            "cl_image_handler(%s) pre_execute kernel(%s) failed",
            XCAM_STR (_name), kernel->get_kernel_name ());

        XCAM_FAIL_RETURN (
            WARNING,
            (ret = kernel->execute ()) == XCAM_RETURN_NO_ERROR,
            ret,
            "cl_image_handler(%s) execute kernel(%s) failed",
            XCAM_STR (_name), kernel->get_kernel_name ());

        XCAM_FAIL_RETURN (
            WARNING,
            (ret = kernel->post_execute ()) == XCAM_RETURN_NO_ERROR,
            ret,
            "cl_image_handler(%s) post_execute kernel(%s) failed",
            XCAM_STR (_name), kernel->get_kernel_name ());
    }

    //CLDevice::instance()->get_context ()->finish ();

    XCAM_OBJ_PROFILING_END (XCAM_STR (_name), 30);

    return XCAM_RETURN_NO_ERROR;
}

};
