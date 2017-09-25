/*
 * cl_3a_image_processor.cpp - CL 3A image processor
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
 * Author: wangfei <feix.w.wang@intel.com>
 */
#include "cl_csc_image_processor.h"
#include "cl_context.h"
#include "cl_csc_handler.h"


namespace XCam {

CLCscImageProcessor::CLCscImageProcessor ()
    : CLImageProcessor ("CLCscImageProcessor")
{
    XCAM_LOG_DEBUG ("CLCscImageProcessor constructed");
}

CLCscImageProcessor::~CLCscImageProcessor ()
{
    XCAM_LOG_DEBUG ("CLCscImageProcessor destructed");
}

XCamReturn
CLCscImageProcessor::create_handlers ()
{
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLContext> context = get_cl_context ();

    XCAM_ASSERT (context.ptr ());

    /* color space conversion */
    image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_YUYVTORGBA);
    _csc = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
    XCAM_FAIL_RETURN (
        WARNING,
        _csc .ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLCscImageProcessor create csc handler failed");

    image_handler->set_pool_type (CLImageHandler::CLVideoPoolType);
    add_handler (image_handler);

    return XCAM_RETURN_NO_ERROR;
}

};
