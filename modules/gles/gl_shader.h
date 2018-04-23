/*
 * gl_shader.h - GL shader
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

#ifndef XCAM_GL_SHADER_H
#define XCAM_GL_SHADER_H

#include <gles/gles_std.h>
#include <map>

namespace XCam {

class GLShader
{
public:
    ~GLShader ();
    static SmartPtr<GLShader> compile_shader (
        GLenum type, const char *src, uint32_t length = -1, const char *name = NULL);

#if 0
    static SmartPtr<GLShader> create_binary_shader (
        GLenum format, const uint8_t *binary, uint32_t length, const char *name = NULL);
#endif

    GLuint get_shader_id () const {
        return _shader_id;
    }
    const char *get_name () const {
        return _name;
    }
    GLenum get_type () const {
        return _shader_type;
    }

private:
    explicit GLShader (GLuint id, GLenum type, const char *name);

private:
    XCAM_DEAD_COPY (GLShader);

private:
    GLenum        _shader_type;
    GLuint        _shader_id;
    char          _name [XCAM_GL_NAME_LENGTH];
};

}

#endif  //XCAM_GL_SHADER_H
