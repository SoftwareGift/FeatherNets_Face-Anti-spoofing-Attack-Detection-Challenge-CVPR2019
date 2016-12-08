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

}

