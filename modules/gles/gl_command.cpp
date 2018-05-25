/*
 * gl_command.cpp - GL command implementation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "gl_command.h"
#include "gl_buffer.h"

namespace XCam {

namespace UniformOps {

template <>
GLenum uniform <GLfloat> (GLint location, GLfloat value)
{
    glUniform1f (location, value);
    return gl_error ();
}

template <>
GLenum uniform <GLint> (GLint location, GLint value)
{
    glUniform1i (location, value);
    return gl_error ();
}

template <>
GLenum uniform <GLuint> (GLint location, GLuint value)
{
    glUniform1ui (location, value);
    return gl_error ();
}

template <>
GLenum uniform_array <GLfloat> (GLint location, const GLfloat *value, GLsizei count)
{
    glUniform1fv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_array <GLint> (GLint location, const GLint *value, GLsizei count)
{
    glUniform1iv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_array <GLuint> (GLint location, const GLuint *value, GLsizei count)
{
    glUniform1uiv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLfloat, 2> (GLint location, const GLfloat *value, GLsizei count)
{
    glUniform2fv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLfloat, 3> (GLint location, const GLfloat *value, GLsizei count)
{
    glUniform3fv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLfloat, 4> (GLint location, const GLfloat *value, GLsizei count)
{
    glUniform4fv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLint, 2> (GLint location, const GLint *value, GLsizei count)
{
    glUniform2iv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLint, 3> (GLint location, const GLint *value, GLsizei count)
{
    glUniform3iv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLint, 4> (GLint location, const GLint *value, GLsizei count)
{
    glUniform4iv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLuint, 2> (GLint location, const GLuint *value, GLsizei count)
{
    glUniform2uiv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLuint, 3> (GLint location, const GLuint *value, GLsizei count)
{
    glUniform3uiv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_vect <GLuint, 4> (GLint location, const GLuint *value, GLsizei count)
{
    glUniform4uiv (location, count, value);
    return gl_error ();
}

template <>
GLenum uniform_mat <GLfloat, 2> (GLint location, const GLfloat *value, GLsizei count)
{
    glUniformMatrix2fv (location, count, GL_FALSE, value);
    return gl_error ();
}

template <>
GLenum uniform_mat <GLfloat, 3> (GLint location, const GLfloat *value, GLsizei count)
{
    glUniformMatrix3fv (location, count, GL_FALSE, value);
    return gl_error ();
}

template <>
GLenum uniform_mat <GLfloat, 4> (GLint location, const GLfloat *value, GLsizei count)
{
    glUniformMatrix4fv (location, count, GL_FALSE, value);
    return gl_error ();
}

}

GLCmdUniform::GLCmdUniform (const GLchar *name)
{
    XCAM_ASSERT (name);
    strncpy (_name, name, sizeof (_name) - 1);
}

GLCmdUniform::~GLCmdUniform ()
{
}

XCamReturn
GLCmdUniform::run (GLuint program)
{
    GLint location = get_uniform_location (program, _name);
    XCAM_FAIL_RETURN (ERROR, location >= 0, XCAM_RETURN_ERROR_UNKNOWN, "get_uniform_location failed");

    GLenum error = uniform (location);
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_UNKNOWN,
        "uniform failed, name:%s, error flag: %s", _name, gl_error_string (error));

    return XCAM_RETURN_NO_ERROR;
}

GLint
GLCmdUniform::get_uniform_location (GLuint program, const GLchar *name)
{
    GLint location = glGetUniformLocation (program, name);
    GLenum error = gl_error ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, -1,
        "get_uniform_location failed, name:%s, error flag: %s",
        XCAM_STR (name), gl_error_string (error));

    XCAM_FAIL_RETURN (
        WARNING, location >= 0, -1,
        "get_uniform_location invalid or unnecessary parameter, name:%s location:%d",
        XCAM_STR (name), location);

    return location;
}

GLCmdBindBufBase::GLCmdBindBufBase (const SmartPtr<GLBuffer> &buf, uint32_t index)
    : _index (index)
{
    XCAM_ASSERT (buf.ptr ());
    _buf = buf;
}

GLCmdBindBufBase::~GLCmdBindBufBase ()
{
}

XCamReturn
GLCmdBindBufBase::run (GLuint program)
{
    XCAM_UNUSED (program);

    XCamReturn ret = _buf->bind_buffer_base (_index);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLCmdBindBufBase failed, idx:%d", _index);

    return XCAM_RETURN_NO_ERROR;
}

GLCmdBindBufRange::GLCmdBindBufRange (const SmartPtr<GLBuffer> &buf, uint32_t index, uint32_t offset_x)
    : _index (index)
    , _offset (offset_x)
    , _size (0)
{
    XCAM_ASSERT (buf.ptr ());
    _buf = buf;

    const GLBufferDesc &desc = buf->get_buffer_desc ();
    _size = desc.size - offset_x;
}

GLCmdBindBufRange::GLCmdBindBufRange (
    const SmartPtr<GLBuffer> &buf, uint32_t index, NV12PlaneIdx plane, uint32_t offset_in_plane)
    : _index (index)
    , _offset (0)
    , _size (0)
{
    XCAM_ASSERT (buf.ptr ());
    _buf = buf;

    const GLBufferDesc &desc = buf->get_buffer_desc ();
    _offset = desc.offsets [plane] + offset_in_plane;
    _size = desc.slice_size [plane] - offset_in_plane;
}

GLCmdBindBufRange::~GLCmdBindBufRange ()
{
}

XCamReturn
GLCmdBindBufRange::run (GLuint program)
{
    XCAM_UNUSED (program);

    XCamReturn ret = _buf->bind_buffer_range (_index, _offset, _size);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLCmdBindBufRange failed, idx:%d", _index);

    return XCAM_RETURN_NO_ERROR;
}

}
