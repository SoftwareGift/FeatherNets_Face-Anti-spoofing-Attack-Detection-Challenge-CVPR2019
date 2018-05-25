/*
 * gl_command.h - GL command class
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

#ifndef XCAM_GL_COMMAND_H
#define XCAM_GL_COMMAND_H

#include <list>
#include <gles/gles_std.h>

namespace XCam {

namespace UniformOps {

template <typename TType>
GLenum uniform (GLint location, TType value);

template <typename TType>
GLenum uniform_array (GLint location, const TType *value, GLsizei count);

template <typename TType, uint32_t TDim>
GLenum uniform_vect (GLint location, const TType *value, GLsizei count = 1);

template <typename TType, uint32_t TDim>
GLenum uniform_mat (GLint location, const TType *value, GLsizei count = 1);
}

class GLCommand
{
public:
    virtual ~GLCommand () {}
    virtual XCamReturn run (GLuint program) = 0;

protected:
    explicit GLCommand () {}

private:
    XCAM_DEAD_COPY (GLCommand);
};
typedef std::list<SmartPtr<GLCommand> > GLCmdList;

class GLCmdUniform
    : public GLCommand
{
public:
    virtual ~GLCmdUniform ();
    virtual XCamReturn run (GLuint program);

protected:
    explicit GLCmdUniform (const GLchar *name);

private:
    GLint get_uniform_location (GLuint program, const GLchar *name);
    virtual GLenum uniform (GLint location) = 0;

protected:
    GLchar        _name[XCAM_GL_NAME_LENGTH];
};

/* uniform single variable */
template <typename TType>
class GLCmdUniformT
    : public GLCmdUniform
{
public:
    GLCmdUniformT (const GLchar *name, TType value)
        : GLCmdUniform (name)
        , _value (value)
    {}
    ~GLCmdUniformT () {}

private:
    virtual GLenum uniform (GLint location) {
        return UniformOps::uniform <TType> (location, _value);
    }

private:
    TType        _value;
};

/* uniform array: TType array[TCount] */
template <typename TType, int TCount>
class GLCmdUniformTArray
    : public GLCmdUniform
{
public:
    GLCmdUniformTArray (const GLchar *name, const TType *value)
        : GLCmdUniform (name)
    {
        XCAM_ASSERT (value);
        memcpy (&_value[0], value, sizeof (TType) * TCount);
    }
    ~GLCmdUniformTArray () {}

private:
    virtual GLenum uniform (GLint location) {
        return UniformOps::uniform_array <TType> (location, _value, TCount);
    }

private:
    TType        _value[TCount];
};

/* uniform vectors: TType vec{TDim}[TCount]*/
template <typename TType, int TDim, int TCount = 1>
class GLCmdUniformTVect
    : public GLCmdUniform
{
public:
    GLCmdUniformTVect (const GLchar *name, const TType *value)
        : GLCmdUniform (name)
    {
        XCAM_ASSERT (value);
        memcpy (&_value[0], value, sizeof (TType) * TDim * TCount);
    }
    ~GLCmdUniformTVect () {}

private:
    virtual GLenum uniform (GLint location) {
        return UniformOps::uniform_vect <TType, TDim> (location, _value, TCount);
    }

private:
    TType        _value[TDim * TCount];
};

/* uniform matrix: TType mat{TColumns}x{TRows}[TCount], only support square matrix */
template <typename TType, int TColumns, int TRows, int TCount = 1>
class GLCmdUniformTMat
    : public GLCmdUniform
{
public:
    GLCmdUniformTMat (const GLchar *name, const TType *value)
        : GLCmdUniform (name)
    {
        XCAM_ASSERT (value);
        memcpy (&_value[0], value, sizeof (TType) * TColumns * TRows * TCount);
    }
    ~GLCmdUniformTMat () {}

private:
    virtual GLenum uniform (GLint location) {
        XCAM_FAIL_RETURN (
            ERROR, TColumns == TRows, -1,
            "uniform_mat only support square matrix, invalid dimension:%dx%d", TColumns, TRows);

        return UniformOps::uniform_mat <TType, TColumns> (location, _value, TCount);
    }

private:
    TType        _value[TColumns * TRows * TCount];
};

class GLBuffer;

class GLCmdBindBufBase
    : public GLCommand
{
public:
    GLCmdBindBufBase (const SmartPtr<GLBuffer> &buf, uint32_t index);
    virtual ~GLCmdBindBufBase ();

    virtual XCamReturn run (GLuint program);

private:
    SmartPtr<GLBuffer>        _buf;
    uint32_t                  _index;
};

class GLCmdBindBufRange
    : public GLCommand
{
public:
    GLCmdBindBufRange (const SmartPtr<GLBuffer> &buf, uint32_t index, uint32_t offset_x = 0);
    GLCmdBindBufRange (
        const SmartPtr<GLBuffer> &buf, uint32_t index, NV12PlaneIdx plane, uint32_t offset_in_plane = 0);
    virtual ~GLCmdBindBufRange ();

    virtual XCamReturn run (GLuint program);

private:
    SmartPtr<GLBuffer>        _buf;
    uint32_t                  _index;
    uint32_t                  _offset;
    uint32_t                  _size;
};

}

#endif // XCAM_GL_COMMAND_H