/*
 * swapped_buffer.cpp - swapped buffer
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

#include <xcam_std.h>
#include "swapped_buffer.h"

namespace XCam {

SwappedBuffer::SwappedBuffer (
    const VideoBufferInfo &info, const SmartPtr<BufferData> &data)
    : BufferProxy (info, data)
    , _swap_flags (SwappedBuffer::SwapNone)
{
    xcam_mem_clear (_swap_offsets);
}

SwappedBuffer::~SwappedBuffer ()
{
}

void
SwappedBuffer::set_swap_info (uint32_t flags, uint32_t* offsets)
{
    _swap_flags = flags;
    XCAM_ASSERT (offsets);
    memcpy(_swap_offsets, offsets, sizeof (_swap_offsets));
}

bool SwappedBuffer::swap_new_buffer_info(
    const VideoBufferInfo &in, uint32_t flags, VideoBufferInfo &out)
{
    out = in;
    if (flags & (uint32_t)(SwapY)) {
        if (in.offsets[0] == _swap_offsets[SwapYOffset0]) {
            out.offsets[0] = _swap_offsets[SwapYOffset1];
        } else {
            XCAM_ASSERT (in.offsets[0] == _swap_offsets[SwapYOffset1]);
            out.offsets[0] = _swap_offsets[SwapYOffset0];
        }
    }
    if (flags & (uint32_t)(SwapUV)) {
        if (in.offsets[1] == _swap_offsets[SwapUVOffset0]) {
            out.offsets[1] = _swap_offsets[SwapUVOffset1];
        } else {
            XCAM_ASSERT (in.offsets[1] == _swap_offsets[SwapUVOffset1]);
            out.offsets[1] = _swap_offsets[SwapUVOffset0];
        }
    }
    return true;
}

SmartPtr<SwappedBuffer>
SwappedBuffer::create_new_swap_buffer (
    const VideoBufferInfo &info, SmartPtr<BufferData> &data)
{
    XCAM_ASSERT (false);
    SmartPtr<SwappedBuffer> out = new SwappedBuffer (info, data);
    return out;
}

SmartPtr<SwappedBuffer>
SwappedBuffer::swap_clone (SmartPtr<SwappedBuffer> self, uint32_t flags)
{
    XCAM_ASSERT (self.ptr () && self.ptr () == (SwappedBuffer*)(this));
    XCAM_FAIL_RETURN(
        WARNING,
        flags && (flags & _swap_flags) == flags,
        NULL,
        "SwappedBuffer swap_clone failed since flags doesn't match");

    const VideoBufferInfo &cur_info = this->get_video_info ();
    VideoBufferInfo out_info;
    XCAM_FAIL_RETURN(
        WARNING,
        swap_new_buffer_info (cur_info, flags, out_info),
        NULL,
        "SwappedBuffer swap_clone failed on out buffer info");

    SmartPtr<BufferData> data = get_buffer_data ();
    XCAM_FAIL_RETURN(
        WARNING,
        data.ptr (),
        NULL,
        "SwappedBuffer swap_clone failed to get buffer data");

    SmartPtr<SwappedBuffer> out = create_new_swap_buffer (out_info, data);
    XCAM_FAIL_RETURN(
        WARNING,
        out.ptr (),
        NULL,
        "SwappedBuffer swap_clone failed to create new swap buffer");
    out->_swap_flags = _swap_flags;
    memcpy (out->_swap_offsets, _swap_offsets, sizeof (_swap_offsets));
    out->set_parent (self);
    return out;
}

};
