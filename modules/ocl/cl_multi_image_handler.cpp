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
CLMultiImageHandler::execute (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
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

    ret = this->prepare_parameters (input, output);
    XCAM_FAIL_RETURN (
        WARNING,
        (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS),
        ret,
        "cl_image_handler (%s) prepare parameters failed", XCAM_STR (_name));
    if (ret == XCAM_RETURN_BYPASS)
        return ret;

    XCAM_OBJ_PROFILING_START;

    for (KernelList::iterator i_kernel = _kernels.begin ();
            i_kernel != _kernels.end (); ++i_kernel) {
        SmartPtr<CLImageKernel> &kernel = *i_kernel;
        XCAM_FAIL_RETURN (WARNING, kernel.ptr(), ret, "kernel empty");

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

        ret = kernel->post_execute (output);
        XCAM_FAIL_RETURN (
            WARNING,
            (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS),
            ret,
            "cl_image_handler(%s) post_execute kernel(%s) failed",
            XCAM_STR (_name), kernel->get_kernel_name ());

        if (ret == XCAM_RETURN_BYPASS)
            break;

        for (HandlerList::iterator i_handler = _handler_list.begin ();
                i_handler != _handler_list.end (); ++i_handler) {
            SmartPtr<CLImageHandler> &sub_handler = *i_handler;
            XCAM_ASSERT (sub_handler.ptr ());

            SmartPtr<CLImageKernel> &sub_handler_last_kernel = *(sub_handler->_kernels.rbegin());
            XCAM_ASSERT (sub_handler_last_kernel.ptr ());
            if (sub_handler_last_kernel.ptr () == kernel.ptr ()) {
                this->sub_handler_execute_done (sub_handler);
                break;
            }
        }
    }

#if ENABLE_PROFILING
    CLDevice::instance()->get_context ()->finish ();
#endif

    XCAM_OBJ_PROFILING_END (XCAM_STR (_name), XCAM_OBJ_DUR_FRAME_NUM);

    if (ret != XCAM_RETURN_NO_ERROR)
        return ret;

    ret = execute_done (output);
    return ret;
}

XCamReturn
CLMultiImageHandler::prepare_parameters (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output)
{
    for (HandlerList::iterator i_handler = _handler_list.begin ();
            i_handler != _handler_list.end (); ++i_handler) {
        SmartPtr<CLImageHandler> &handler = *i_handler;
        XCAM_ASSERT (handler.ptr ());

        XCamReturn ret = handler->prepare_parameters (input, output);
        if (ret == XCAM_RETURN_BYPASS)
            return ret;

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
CLMultiImageHandler::execute_done (SmartPtr<DrmBoBuffer> &output)
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

