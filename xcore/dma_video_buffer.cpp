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

class DmaVideoBufferPriv
    : public DmaVideoBuffer
{
    friend SmartPtr<DmaVideoBuffer> external_buf_to_dma_buf (XCamVideoBuffer *buf);
protected:
    DmaVideoBufferPriv (const VideoBufferInfo &info, XCamVideoBuffer *buf);
    ~DmaVideoBufferPriv ();

private:
    XCamVideoBuffer *_external_buf;
};

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

DmaVideoBufferPriv::DmaVideoBufferPriv (const VideoBufferInfo &info, XCamVideoBuffer *buf)
    : DmaVideoBuffer (info, xcam_video_buffer_get_fd (buf), false)
    , _external_buf (buf)
{
    if (buf->ref)
        xcam_video_buffer_ref (buf);
}

DmaVideoBufferPriv::~DmaVideoBufferPriv ()
{
    if (_external_buf && _external_buf->unref && _external_buf->ref)
        xcam_video_buffer_unref (_external_buf);
}

SmartPtr<DmaVideoBuffer>
external_buf_to_dma_buf (XCamVideoBuffer *buf)
{
    VideoBufferInfo buf_info;
    SmartPtr<DmaVideoBuffer> video_buffer;

    XCAM_FAIL_RETURN (
        ERROR, buf, NULL,
        "external_buf_to_dma_buf failed since buf is NULL");

    int buffer_fd = 0;
    if (buf->get_fd)
        buffer_fd = xcam_video_buffer_get_fd(buf);

    XCAM_FAIL_RETURN (
        ERROR, buffer_fd > 0, NULL,
        "external_buf_to_dma_buf failed, can't get buf file-handle");

    buf_info.init (buf->info.format, buf->info.width, buf->info.height,
                   buf->info.aligned_width, buf->info.aligned_height, buf->info.size);
    video_buffer = new DmaVideoBufferPriv (buf_info, buf);
    XCAM_ASSERT (video_buffer.ptr ());
    return video_buffer;
}

}
