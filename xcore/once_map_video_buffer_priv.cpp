/*
 * once_map_video_buffer_priv.cpp
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include <xcam_std.h>
#include <video_buffer.h>

namespace XCam {

class OnceMapVideoBuffer
    : public VideoBuffer
{
public:
    OnceMapVideoBuffer (const VideoBufferInfo &info, uint8_t* buffer);

    virtual ~OnceMapVideoBuffer ();

    virtual uint8_t *map ();
    virtual bool unmap ();
    virtual int get_fd ();

private:

    XCAM_DEAD_COPY (OnceMapVideoBuffer);

private:
    uint8_t* _buffer;
};

OnceMapVideoBuffer::OnceMapVideoBuffer (const VideoBufferInfo &info, uint8_t* buffer)
    : VideoBuffer (info)
    , _buffer (buffer)
{
    XCAM_ASSERT (buffer != NULL);
}

OnceMapVideoBuffer::~OnceMapVideoBuffer ()
{
}

uint8_t *
OnceMapVideoBuffer::map ()
{
    return _buffer;
}

bool
OnceMapVideoBuffer::unmap ()
{
    return true;
}

int
OnceMapVideoBuffer::get_fd ()
{
    XCAM_ASSERT (false && "OnceMapVideoBuffer::get_fd not supported");
    return -1;
}

SmartPtr<VideoBuffer>
external_buf_to_once_map_buf (
    uint8_t* buf, uint32_t format,
    uint32_t width, uint32_t height,
    uint32_t aligned_width, uint32_t aligned_height,
    uint32_t size)
{
    VideoBufferInfo buf_info;
    SmartPtr<OnceMapVideoBuffer> video_buffer;

    XCAM_FAIL_RETURN (
        ERROR, buf, NULL,
        "external_buf_to_map_buf failed since buf is NULL");

    buf_info.init (format, width, height,
                   aligned_width, aligned_height, size);
    video_buffer = new OnceMapVideoBuffer (buf_info, buf);
    XCAM_ASSERT (video_buffer.ptr ());
    return video_buffer;
}

}
