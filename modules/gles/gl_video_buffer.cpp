/*
 * gl_video_buffer.cpp - GL video buffer implementation
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "gl_video_buffer.h"

namespace XCam {

class GLVideoBufferData
    : public BufferData
{
public:
    explicit GLVideoBufferData (SmartPtr<GLBuffer> &body);
    ~GLVideoBufferData ();

    virtual uint8_t *map ();
    virtual bool unmap ();

    SmartPtr<GLBuffer> &get_buf () {
        return _buf;
    }

private:
    uint8_t              *_buf_ptr;
    SmartPtr<GLBuffer>    _buf;
};

GLVideoBufferData::GLVideoBufferData (SmartPtr<GLBuffer> &body)
    : _buf_ptr (NULL)
    , _buf (body)
{
    XCAM_ASSERT (body.ptr ());
}

GLVideoBufferData::~GLVideoBufferData ()
{
    unmap ();
    _buf.release ();
}

uint8_t *
GLVideoBufferData::map ()
{
    if (_buf_ptr)
        return _buf_ptr;

    uint32_t size = _buf->get_size ();
    _buf_ptr = (uint8_t *) _buf->map_range (0, size, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, _buf_ptr, NULL, "GLVideoBufferData map data failed");

    return _buf_ptr;
}

bool
GLVideoBufferData::unmap ()
{
    if (!_buf_ptr)
        return true;

    XCamReturn ret = _buf->unmap ();
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, false,
        "GLVideoBufferData unmap data failed");

    _buf_ptr = NULL;
    return true;
}

GLVideoBuffer::GLVideoBuffer (const VideoBufferInfo &info, const SmartPtr<BufferData> &data)
    : BufferProxy (info, data)
{
}

SmartPtr<GLBuffer>
GLVideoBuffer::get_gl_buffer ()
{
    SmartPtr<BufferData> data = get_buffer_data ();
    SmartPtr<GLVideoBufferData> gl_data = data.dynamic_cast_ptr<GLVideoBufferData> ();
    XCAM_FAIL_RETURN (
        WARNING, gl_data.ptr (), NULL,
        "GLVideoBuffer get_buffer_data failed with NULL");

    return gl_data->get_buf ();
}

GLVideoBufferPool::GLVideoBufferPool ()
    : _target (GL_SHADER_STORAGE_BUFFER)
{
}

GLVideoBufferPool::GLVideoBufferPool (const VideoBufferInfo &info, GLenum target)
    : _target (target)
{
    set_video_info (info);
}

GLVideoBufferPool::~GLVideoBufferPool ()
{
}

SmartPtr<BufferData>
GLVideoBufferPool::allocate_data (const VideoBufferInfo &info)
{
    SmartPtr<GLBuffer> buf =
        XCam::GLBuffer::create_buffer (_target, NULL, info.size, GL_STATIC_DRAW);
    XCAM_ASSERT (buf.ptr ());

    return new GLVideoBufferData (buf);
}

SmartPtr<BufferProxy>
GLVideoBufferPool::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    XCAM_ASSERT (data.ptr ());

    const VideoBufferInfo &info = get_video_info ();
    SmartPtr<GLVideoBuffer> buf = new GLVideoBuffer (info, data);
    XCAM_ASSERT (buf.ptr ());

    return buf;
}

};
