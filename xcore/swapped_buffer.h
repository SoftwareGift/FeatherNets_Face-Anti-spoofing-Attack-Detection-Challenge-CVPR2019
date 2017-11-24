/*
 * swapped_buffer.h - swapped buffer
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

#ifndef XCAM_SWAPPED_BUFFER_H
#define XCAM_SWAPPED_BUFFER_H

#include <xcam_std.h>
#include <buffer_pool.h>

namespace XCam {

class SwappedBuffer
    : public virtual BufferProxy
{
public:
    enum SwapFlags {
        SwapNone = 0,
        SwapY    = 1,
        SwapUV   = 2,
    };

    enum SwapOffsets {
        SwapYOffset0 = 0,
        SwapYOffset1 = 1,
        SwapUVOffset0 = 2,
        SwapUVOffset1 = 3,
    };

    enum InitOrder {
        OrderYMask = 0x000F,
        OrderY0Y1 = 0x0001,
        OrderY1Y0 = 0x0002,
        OrderUVMask = 0x0F00,
        OrderUV0UV1 = 0x0100,
        OrderUV1UV0 = 0x0200,
    };

protected:
    explicit SwappedBuffer (
        const VideoBufferInfo &info, const SmartPtr<BufferData> &data);

public:
    virtual ~SwappedBuffer ();
    void set_swap_info (uint32_t flags, uint32_t* offsets);

    SmartPtr<SwappedBuffer> swap_clone (
        SmartPtr<SwappedBuffer> self, uint32_t flags);

protected:
    virtual SmartPtr<SwappedBuffer> create_new_swap_buffer (
        const VideoBufferInfo &info, SmartPtr<BufferData> &data);

    bool swap_new_buffer_info (
        const VideoBufferInfo &in, uint32_t flags, VideoBufferInfo &out);

private:
    XCAM_DEAD_COPY (SwappedBuffer);

protected:
    uint32_t                   _swap_flags;
    uint32_t                   _swap_offsets[XCAM_VIDEO_MAX_COMPONENTS * 2];
};

}

#endif //XCAM_SWAPPED_BUFFER_H
