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

namespace XCam {

#define XCAM_CL_IMAGE_HANDLER_DEFAULT_BUF_NUM 6

CLImageKernel::CLImageKernel (SmartPtr<CLContext> &context, const char *name)
    : CLKernel (context, name)
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
    uint32_t dim = XCAM_DEFAULT_IMAGE_DIM;
    size_t global[XCAM_DEFAULT_IMAGE_DIM] = {0};
    size_t local[XCAM_DEFAULT_IMAGE_DIM] = {1, 1};
    cl_mem mem0 = NULL, mem1 = NULL;


    _image_in = new CLVaImage (context, input);
    _image_out = new CLVaImage (context, output);

    XCAM_ASSERT (_image_in->is_valid () && _image_out->is_valid ());
    XCAM_FAIL_RETURN (
        WARNING,
        _image_in->is_valid () && _image_out->is_valid (),
        XCAM_RETURN_ERROR_MEM,
        "cl image kernel(%s) in/out memory not available", get_kernel_name ());

    //set args;
    mem0 = _image_in->get_mem_id ();
    ret = set_argument (0, &mem0, sizeof (mem0));
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "cl image kernel(%s) set argc(0) failed", get_kernel_name ());

    mem1 = _image_out->get_mem_id ();
    ret = set_argument (1, &mem1, sizeof (mem1));
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "cl image kernel(%s) set argc(1) failed", get_kernel_name ());

    {
        const cl_libva_image &out_info = _image_out->get_image_info ();
        global[0] = out_info.row_pitch / 4;
        global[1] = out_info.height / 4;
    }
    ret = set_work_size (dim, global, local);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "cl image kernel(%s) set work size failed", get_kernel_name ());

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

XCamReturn
CLImageHandler::ensure_buffer_pool (const VideoBufferInfo &video_info)
{
    SmartPtr<DrmBoBufferPool> buffer_pool;
    SmartPtr<DrmDisplay> display;

    if (_buf_pool.ptr ())
        return XCAM_RETURN_NO_ERROR;

    display = DrmDisplay::instance ();
    XCAM_FAIL_RETURN(
        WARNING,
        display.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLImageHandler(%s) failed to get drm dispay", XCAM_STR (_name));

    buffer_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (buffer_pool.ptr ());
    buffer_pool->set_buffer_info (video_info);

    XCAM_FAIL_RETURN(
        WARNING,
        buffer_pool->init (XCAM_CL_IMAGE_HANDLER_DEFAULT_BUF_NUM),
        XCAM_RETURN_ERROR_CL,
        "CLImageHandler(%s) failed to init drm buffer pool", XCAM_STR (_name));

    _buf_pool = buffer_pool;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageHandler::prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    SmartPtr<DrmBoBuffer> new_buf;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    ret = ensure_buffer_pool (input->get_video_info ());
    XCAM_FAIL_RETURN(
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "CLImageHandler(%s) ensure drm buffer pool failed", XCAM_STR (_name));

    new_buf = _buf_pool->get_buffer (_buf_pool);
    XCAM_FAIL_RETURN(
        WARNING,
        new_buf.ptr(),
        XCAM_RETURN_ERROR_UNKNOWN,
        "CLImageHandler(%s) failed to get drm buffer from pool", XCAM_STR (_name));

    output = new_buf;
    return XCAM_RETURN_NO_ERROR;
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

    return XCAM_RETURN_NO_ERROR;
}

};
