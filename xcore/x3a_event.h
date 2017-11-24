/*
 * x3a_event.h - event
 *
 *  Copyright (c) 2014 Intel Corporation
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

#ifndef XCAM_3A_EVENT_H
#define XCAM_3A_EVENT_H

#include <xcam_std.h>

namespace XCam {

class X3aEvent
//:public ObjectLife
{
public:
    enum Type {
        TYPE_ISP_STATISTICS,
        TYPE_ISP_FRAME_SYNC,
    };

protected:
    explicit X3aEvent (X3aEvent::Type type, uint64_t timestamp)
        : _timestamp (timestamp)
        , _type (type)
    {}
    virtual ~X3aEvent() {}

public:
    uint64_t get_timestamp () const {
        return _timestamp;
    }
    Type get_type () const {
        return _type;
    }

private:
    XCAM_DEAD_COPY (X3aEvent);

protected:
    uint64_t       _timestamp;
    X3aEvent::Type _type;
};

};

#endif //XCAM_3A_EVENT_H

