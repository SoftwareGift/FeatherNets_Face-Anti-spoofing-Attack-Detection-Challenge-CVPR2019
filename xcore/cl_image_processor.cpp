/*
 * cl_image_processor.cpp - CL image processor
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
#include "cl_image_processor.h"
#include "cl_context.h"
#include "cl_device.h"
#include "cl_image_handler.h"
#include "drm_display.h"
#include "cl_demo_handler.h"
#include "cl_blc_handler.h"
#include "cl_csc_handler.h"


namespace XCam {

CLImageProcessor::CLImageProcessor (const char* name)
    : ImageProcessor (name ? name : "CLImageProcessor")
{
    _context = CLDevice::instance ()->get_context ();
    XCAM_ASSERT (_context.ptr());
    XCAM_LOG_DEBUG ("CLImageProcessor constructed");
    XCAM_OBJ_PROFILING_INIT;
}

CLImageProcessor::~CLImageProcessor ()
{
    XCAM_LOG_DEBUG ("CLImageProcessor destructed");
}

bool
CLImageProcessor::add_handler (SmartPtr<CLImageHandler> &handler)
{
    XCAM_ASSERT (handler.ptr ());
    _handlers.push_back (handler);
    return true;
}

SmartPtr<CLContext>
CLImageProcessor::get_cl_context ()
{
    return _context;
}

bool
CLImageProcessor::can_process_result (SmartPtr<X3aResult> &result)
{
    XCAM_UNUSED (result);
    return false;
}

XCamReturn
CLImageProcessor::apply_3a_results (X3aResultList &results)
{
    XCAM_UNUSED (results);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageProcessor::apply_3a_result (SmartPtr<X3aResult> &result)
{
    XCAM_UNUSED (result);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLImageProcessor::process_buffer (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    SmartPtr<DrmBoBuffer> drm_bo_in, drm_bo_out;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<DrmDisplay> display = DrmDisplay::instance ();

    XCAM_OBJ_PROFILING_START;

    drm_bo_in = display->convert_to_drm_bo_buf (display, input);
    XCAM_FAIL_RETURN (
        WARNING,
        drm_bo_in.ptr (),
        XCAM_RETURN_ERROR_MEM,
        "CL image processor can't handle this buffer, maybe type error");

    STREAM_LOCK;

    if (_handlers.empty()) {
        ret = create_handlers ();
    }

    for (ImageHandlerList::iterator i_handler = _handlers.begin ();
            i_handler != _handlers.end ();  ++i_handler)
    {
        ret = (*i_handler)->execute (drm_bo_in, drm_bo_out);
        XCAM_FAIL_RETURN (
            WARNING,
            ret == XCAM_RETURN_NO_ERROR,
            ret,
            "CL image handler(%s) execute buffer failed", (*i_handler)->get_name());
        drm_bo_in = drm_bo_out;
    }

    if (drm_bo_out.ptr ())
        drm_bo_out->clear_attached_buffers ();

    XCAM_OBJ_PROFILING_END(get_name(), 30);

    output = drm_bo_out;
    return XCAM_RETURN_NO_ERROR;
}

void
CLImageProcessor::emit_stop ()
{
    for (ImageHandlerList::iterator i_handler = _handlers.begin ();
            i_handler != _handlers.end ();  ++i_handler) {
        (*i_handler)->emit_stop ();
    }
}

XCamReturn
CLImageProcessor::create_handlers ()
{
    SmartPtr<CLImageHandler> demo_handler;
    demo_handler = create_cl_demo_image_handler (_context);
    XCAM_FAIL_RETURN (
        WARNING,
        demo_handler.ptr (),
        XCAM_RETURN_ERROR_CL,
        "CLImageProcessor create demo handler failed");
    add_handler (demo_handler);

    return XCAM_RETURN_NO_ERROR;
}

};
