/*
 * cl_event.cpp - CL event
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

#include "cl_event.h"

namespace XCam {

SmartPtr<CLEvent>  CLEvent::NullEvent;
CLEventList CLEvent::EmptyList;

CLEvent::CLEvent (cl_event event_id)
    : _event_id (event_id)
{
}

CLEvent::~CLEvent ()
{
    if (_event_id) {
        clReleaseEvent (_event_id);
    }
}

XCamReturn
CLEvent::wait ()
{
    cl_int error_code = CL_SUCCESS;

    XCAM_FAIL_RETURN (
        DEBUG,
        _event_id,
        XCAM_RETURN_ERROR_PARAM,
        "cl event wait failed, there's no event id");

    error_code = clWaitForEvents (1, &_event_id);

    XCAM_FAIL_RETURN (
        WARNING,
        error_code == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "cl event wait failed with error cod:%d", error_code);

    return XCAM_RETURN_NO_ERROR;
}

bool
CLEvent::get_cl_event_info (
    cl_event_info param_name, size_t param_size,
    void *param, size_t *param_size_ret)
{
    cl_int error_code = CL_SUCCESS;

    XCAM_FAIL_RETURN (
        DEBUG,
        _event_id,
        false,
        "cl event wait failed, there's no event id");

    clGetEventInfo (_event_id, param_name, param_size, param, param_size_ret);

    XCAM_FAIL_RETURN(
        WARNING,
        error_code == CL_SUCCESS,
        false,
        "clGetEventInfo failed on param:%d, errno:%d", param_name, error_code);
    return true;
}

XCamReturn
cl_events_wait (CLEventList &event_list)
{
#define XCAM_MAX_CL_EVENT_COUNT 256

    cl_event event_ids [XCAM_MAX_CL_EVENT_COUNT];
    uint32_t event_count = 0;
    cl_int error_code = CL_SUCCESS;

    if (event_list.empty ())
        return XCAM_RETURN_NO_ERROR;

    xcam_mem_clear (event_ids);
    for (CLEventList::iterator iter = event_list.begin ();
            iter != event_list.end (); ++iter) {
        SmartPtr<CLEvent> &event = *iter;
        XCAM_ASSERT (event->get_event_id ());
        event_ids[event_count++] = event->get_event_id ();
        if (event_count >= XCAM_MAX_CL_EVENT_COUNT)
            break;
    }

    XCAM_ASSERT (event_count > 0);

    error_code = clWaitForEvents (event_count, event_ids);

    XCAM_FAIL_RETURN (
        WARNING,
        error_code == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "cl events wait failed with error cod:%d", error_code);

    return XCAM_RETURN_NO_ERROR;
}

};
