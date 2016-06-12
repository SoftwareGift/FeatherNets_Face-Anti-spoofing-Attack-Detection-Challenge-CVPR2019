/*
 * priority_buffer_queue.cpp - priority buffer queue
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

#include "priority_buffer_queue.h"

#define XCAM_PRIORITY_BUFFER_FIXED_DELAY 8

namespace XCam {

bool
PriorityBuffer::priority_greater_than (const PriorityBuffer& buf) const
{
    int32_t result =
        ((int32_t)(buf.seq_num - this->seq_num) * XCAM_PRIORITY_BUFFER_FIXED_DELAY +
         (int32_t)(buf.rank - this->rank));
    if (result == 0) {
        return (int32_t)(buf.seq_num - this->seq_num) > 0;
    }
    return result > 0;
}


bool
PriorityBufferQueue::push_priority_buf (const SmartPtr<PriorityBuffer> &buf)
{
    XCAM_ASSERT (buf.ptr ());
    SmartLock lock (_mutex);

    ObjList::iterator iter = _obj_list.begin ();

    for (; iter != _obj_list.end (); ++iter) {
        SmartPtr<PriorityBuffer> &current = *iter;
        XCAM_ASSERT (current.ptr ());
        if (buf->priority_greater_than (*current.ptr()))
            break;
    }

    _obj_list.insert (iter, buf);
    _new_obj_cond.signal ();
    return true;
}

};
