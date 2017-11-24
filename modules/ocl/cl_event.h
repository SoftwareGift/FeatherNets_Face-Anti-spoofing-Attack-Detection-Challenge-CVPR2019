/*
 * cl_event.h - CL event
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

#ifndef XCAM_CL_EVENT_H
#define XCAM_CL_EVENT_H

#include <xcam_std.h>
#include <list>
#include <CL/cl.h>

namespace XCam {

class CLEvent;

typedef std::list<SmartPtr<CLEvent>> CLEventList;

class CLEvent {
public:
    explicit CLEvent (cl_event event_id = NULL);
    ~CLEvent ();
    void set_event_id (cl_event event_id) {
        _event_id = event_id;
    }
    cl_event &get_event_id () {
        return _event_id;
    }

    XCamReturn wait ();

    bool get_cl_event_info (
        cl_event_info param_name, size_t param_size,
        void *param, size_t *param_size_ret = NULL);

private:

    XCAM_DEAD_COPY (CLEvent);

public:
    static SmartPtr<CLEvent>  NullEvent;
    static CLEventList EmptyList;

private:
    cl_event  _event_id;
};

XCamReturn
cl_events_wait (CLEventList &event_list);
};

#endif //XCAM_CL_EVENT_H