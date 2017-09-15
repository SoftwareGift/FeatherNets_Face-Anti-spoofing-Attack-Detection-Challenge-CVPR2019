/*
 * cl_multi_image_handler.cpp - CL multi-image handler
 *
 *  Copyright (c) 2016 Intel Corporation
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

#include "cl_multi_image_handler.h"
#if ENABLE_PROFILING
#include "cl_device.h"
#endif

namespace XCam {

CLMultiImageHandler::CLMultiImageHandler (const SmartPtr<CLContext> &context, const char *name)
    : CLImageHandler (context, name)
{
}

CLMultiImageHandler::~CLMultiImageHandler ()
{
    _handler_list.clear ();
}

bool
CLMultiImageHandler::add_image_handler (SmartPtr<CLImageHandler> &handler)
{
    _handler_list.push_back (handler);
    return append_kernels (handler);
}

XCamReturn
CLMultiImageHandler::execute_kernels ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    for (KernelList::iterator i_kernel = _kernels.begin ();
            i_kernel != _kernels.end (); ++i_kernel) {
        SmartPtr<CLImageKernel> &kernel = *i_kernel;
        XCAM_FAIL_RETURN (WARNING, kernel.ptr(), ret, "kernel empty");

        if (!kernel->is_enabled ())
            continue;

        ret = execute_kernel (kernel);

        XCAM_FAIL_RETURN (
            WARNING,
            (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS), ret,
            "cl_multi_image_handler(%s) execute kernel(%s) failed",
            XCAM_STR (_name), kernel->get_kernel_name ());

        if (ret != XCAM_RETURN_NO_ERROR)
            break;

        for (HandlerList::iterator i_handler = _handler_list.begin ();
                i_handler != _handler_list.end (); ++i_handler) {
            SmartPtr<CLImageHandler> &sub_handler = *i_handler;
            XCAM_ASSERT (sub_handler.ptr ());

            SmartPtr<CLImageKernel> &sub_handler_last_kernel = *(sub_handler->_kernels.rbegin());
            XCAM_ASSERT (sub_handler_last_kernel.ptr ());
            if (sub_handler_last_kernel.ptr () == kernel.ptr ()) {
                sub_handler->reset_buf_cache (NULL, NULL);
                sub_handler_execute_done (sub_handler);
                break;
            }
        }
    }

    return ret;
}

XCamReturn
CLMultiImageHandler::ensure_handler_parameters (
    const SmartPtr<CLImageHandler> &handler, SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    XCAM_ASSERT (handler.ptr ());
    return handler->ensure_parameters (input, output);
}

XCamReturn
CLMultiImageHandler::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    for (HandlerList::iterator i_handler = _handler_list.begin ();
            i_handler != _handler_list.end (); ++i_handler) {
        SmartPtr<CLImageHandler> &handler = *i_handler;
        XCAM_ASSERT (handler.ptr ());
        XCamReturn ret = ensure_handler_parameters (handler, input, output);

        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            ret,
            "CLMultiImageHandler(%s) prepare parameters failed on handler(%s)",
            XCAM_STR (get_name ()), XCAM_STR (handler->get_name ()));
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLMultiImageHandler::execute_done (SmartPtr<VideoBuffer> &output)
{
    for (HandlerList::iterator i_handler = _handler_list.begin ();
            i_handler != _handler_list.end (); ++i_handler) {
        SmartPtr<CLImageHandler> &handler = *i_handler;
        XCAM_ASSERT (handler.ptr ());

        XCamReturn ret = handler->execute_done (output);
        if (ret == XCAM_RETURN_BYPASS)
            return ret;

        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            ret,
            "CLMultiImageHandler(%s) execute buffer done failed on handler(%s)",
            XCAM_STR (get_name ()), XCAM_STR (handler->get_name ()));
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLMultiImageHandler::sub_handler_execute_done (SmartPtr<CLImageHandler> &handler)
{
    XCAM_UNUSED (handler);
    return XCAM_RETURN_NO_ERROR;
}

}

