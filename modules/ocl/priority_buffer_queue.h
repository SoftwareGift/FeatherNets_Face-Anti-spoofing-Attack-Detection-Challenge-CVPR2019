/*
 * priority_buffer_queue.h - priority buffer queue
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

#ifndef XCAM_PRIORITY_BUFFER_QUEUE_H
#define XCAM_PRIORITY_BUFFER_QUEUE_H

#include <xcam_std.h>
#include <safe_list.h>
#include <ocl/cl_image_handler.h>

namespace XCam {

struct PriorityBuffer
{
    SmartPtr<VideoBuffer>     data;
    SmartPtr<CLImageHandler>  handler;
    uint32_t                  rank;
    uint32_t                  seq_num;

public:
    PriorityBuffer ()
        : rank (0)
        , seq_num (0)
    {}

    void set_seq_num (const uint32_t value) {
        seq_num = value;
    }
    uint32_t get_seq_num () const {
        return seq_num;
    }

    // when change to next rank
    void down_rank () {
        ++rank;
    }

    bool priority_greater_than (const PriorityBuffer& buf) const;
};

class PriorityBufferQueue
    : public SafeList<PriorityBuffer>
{
public:

    PriorityBufferQueue () {}
    ~PriorityBufferQueue () {}

    bool push_priority_buf (const SmartPtr<PriorityBuffer> &buf);

private:
    XCAM_DEAD_COPY (PriorityBufferQueue);
};

};

#endif //XCAM_PRIORITY_BUFFER_QUEUE_H
