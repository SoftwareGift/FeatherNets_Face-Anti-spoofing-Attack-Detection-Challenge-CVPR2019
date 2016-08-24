/*
 * dma_video_buffer.cpp - dma buffer
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

#include "xcam_utils.h"
#include "dma_video_buffer.h"

namespace XCam {

DmaVideoBuffer::DmaVideoBuffer (const VideoBufferInfo &info, int dma_fd, bool need_close_fd)
    : VideoBuffer (info)
    , _dma_fd (dma_fd)
    , _need_close_fd (need_close_fd)
{
    XCAM_ASSERT (dma_fd >= 0);
}

DmaVideoBuffer::~DmaVideoBuffer ()
{
    if (_need_close_fd && _dma_fd > 0)
        close (_dma_fd);
}

uint8_t *
DmaVideoBuffer::map ()
{
    XCAM_ASSERT (false && "DmaVideoBuffer::map not supported");
    return NULL;
}
bool
DmaVideoBuffer::unmap ()
{
    XCAM_ASSERT (false && "DmaVideoBuffer::map not supported");
    return false;
}

int
DmaVideoBuffer::get_fd ()
{
    return _dma_fd;
}

}
