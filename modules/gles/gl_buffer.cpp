/*
 * gl_buffer.cpp - GL buffer
 *
 *  Copyright (c) 2018 Intel Corporation
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

#include "gl_buffer.h"

namespace XCam {

GLBufferDesc::GLBufferDesc ()
    : format (V4L2_PIX_FMT_NV12)
    , width (0)
    , height (0)
    , aligned_width (0)
    , aligned_height (0)
    , size (0)
{
    xcam_mem_clear (strides);
    xcam_mem_clear (slice_size);
    xcam_mem_clear (offsets);
}

GLBuffer::MapRange::MapRange ()
    : offset (0)
    , len (0)
    , flags (0)
    , ptr (0)
{
}

void
GLBuffer::MapRange::clear ()
{
    offset = 0;
    len = 0;
    flags = 0;
    ptr = NULL;
}

bool
GLBuffer::MapRange::is_mapped () const
{
    return ptr;
}

GLBuffer::GLBuffer (GLuint id, GLenum target, GLenum usage, uint32_t size)
    : _target (target)
    , _usage (usage)
    , _buf_id (id)
    , _size (size)
{
}

XCamReturn
GLBuffer::bind ()
{
    glBindBuffer (_target, _buf_id);
    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GL bind buffer:%d failed. error:%d", _buf_id, error);
    return XCAM_RETURN_NO_ERROR;
}

GLBuffer::~GLBuffer ()
{
    if (_buf_id) {
        glDeleteBuffers (1, &_buf_id);

        GLenum error = glGetError ();
        if (error != GL_NO_ERROR) {
            XCAM_LOG_WARNING (
                "GL Buffer delete buffer failed, error_no:%d", error);
        }
    }
}

SmartPtr<GLBuffer>
GLBuffer::create_buffer (
    GLenum target,
    const GLvoid *data, uint32_t size,
    GLenum usage)
{
    XCAM_ASSERT (size > 0);

    GLuint buf_id = 0;
    glGenBuffers (1, &buf_id);
    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, buf_id && (error == GL_NO_ERROR), NULL,
        "GL buffer creation failed. error:%d", error);

    glBindBuffer (target, buf_id);
    XCAM_FAIL_RETURN (
        ERROR, (error = glGetError ()) == GL_NO_ERROR, NULL,
        "GL buffer creation failed when bind buffer:%d. error:%d",
        buf_id, error);

    glBufferData (target, size, data, usage);
    XCAM_FAIL_RETURN (
        ERROR, (error = glGetError ()) == GL_NO_ERROR, NULL,
        "GL buffer creation failed in glBufferData, id:%d. error:%d",
        buf_id, error);

    SmartPtr<GLBuffer> buf_obj =
        new GLBuffer (buf_id, target, usage, size);

    return buf_obj;
}

void *
GLBuffer::map_range (uint32_t offset, uint32_t length, GLbitfield flags)
{
    if (length == 0)
        length = _size;

    if (_mapped_range.is_mapped () &&
            _mapped_range.flags == flags &&
            _mapped_range.offset == offset &&
            _mapped_range.len == length) {
        return _mapped_range.ptr;
    }
    _mapped_range.clear ();

    XCamReturn ret = bind ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), NULL,
        "GL bind buffer failed, buf_id:%d", _buf_id);

    void *ptr = glMapBufferRange (_target, offset, length, flags);
    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, ptr && (error == GL_NO_ERROR), NULL,
        "GL buffer map range failed, buf_id:%d, offset:%d, len:%d, flags:%d, error:%d",
        _buf_id, offset, length, flags, error);

    _mapped_range.offset = offset;
    _mapped_range.len = length;
    _mapped_range.flags = flags;
    _mapped_range.ptr = ptr;

    return ptr;
}

XCamReturn
GLBuffer::flush_map ()
{
    if (!_mapped_range.is_mapped ())
        return XCAM_RETURN_ERROR_ORDER;

    XCAM_FAIL_RETURN (
        ERROR, _mapped_range.flags & GL_MAP_FLUSH_EXPLICIT_BIT,
        XCAM_RETURN_ERROR_GLES,
        "GL buffer flush_map buf:%d failed, invalid flags(:%d)",
        _buf_id, _mapped_range.flags);

    XCamReturn ret = bind ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GL bind buffer failed, buf_id:%d", _buf_id);

    glFlushMappedBufferRange (_target, _mapped_range.offset,  _mapped_range.len);
    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR,
        XCAM_RETURN_ERROR_GLES,
        "GL buffer flush_map buf:%d failed, error:%d",
        _buf_id, error);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLBuffer::unmap ()
{
    if (!_mapped_range.is_mapped ())
        return XCAM_RETURN_ERROR_ORDER;

    XCamReturn ret = bind ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GL bind buffer failed, buf_id:%d", _buf_id);

    XCAM_FAIL_RETURN (
        ERROR, glUnmapBuffer (_target), XCAM_RETURN_ERROR_GLES,
        "GL buffer unmap buf:%d failed, error:%d",
        _buf_id, glGetError ());

    _mapped_range.clear ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLBuffer::bind_buffer_base (uint32_t index)
{
    XCamReturn ret = bind ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GL bind buffer failed, buf_id:%d", _buf_id);

    glBindBufferBase (_target, index, _buf_id);
    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GL bind buffer base failed. buf_id:%d failed, idx:%d, error:%d",
        _buf_id, index, error);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLBuffer::bind_buffer_range (uint32_t index, uint32_t offset, uint32_t size)
{
    XCamReturn ret = bind ();
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GL bind buffer failed, buf_id:%d", _buf_id);

    glBindBufferRange (_target, index, _buf_id, offset, size);
    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GL bind buffer range failed. buf_id:%d failed, idx:%d, error:%d",
        _buf_id, index, error);

    return XCAM_RETURN_NO_ERROR;
}

}

