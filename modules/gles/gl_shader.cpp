/*
 * gl_shader.cpp - GL shader
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

#include "gl_shader.h"

namespace XCam {

GLShader::GLShader (GLuint id, GLenum type, const char *name)
    : _shader_type (type)
    , _shader_id (id)
{
    XCAM_ASSERT (name);
    strncpy (_name, name, sizeof (_name) - 1);
}

GLShader::~GLShader ()
{
    if (_shader_id) {
        glDeleteShader (_shader_id);

        GLenum error = glGetError ();
        if (error != GL_NO_ERROR) {
            XCAM_LOG_WARNING (
                "GL Shader delete shader failed, error_no:%d", error);
        }
    }
}

SmartPtr<GLShader>
GLShader::compile_shader (const GLShaderInfo &info)
{
    GLuint shader_id = glCreateShader (info.type);
    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, shader_id && (error == GL_NO_ERROR), NULL,
        "GL create shader(:%s) failed, error:%d.",
        XCAM_STR (info.name), error);

    GLint tmp_len = info.len ? info.len : strlen (info.src);
    glShaderSource (shader_id, 1, &info.src, &tmp_len);
    XCAM_FAIL_RETURN (
        ERROR, (error = glGetError ()) == GL_NO_ERROR, NULL,
        "GL create shader(:%s) failed in source loading, error:%d.",
        XCAM_STR (info.name), error);

    glCompileShader (shader_id);
    error = glGetError ();

    GLint status;
    glGetShaderiv (shader_id, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        GLint length;
        std::vector<char> compile_log;
        glGetShaderiv (shader_id, GL_INFO_LOG_LENGTH, &length);
        compile_log.resize (length + 1);
        glGetShaderInfoLog (shader_id, length, &length, &compile_log[0]);
        XCAM_LOG_ERROR (
            "GL create sharder(:%s) compile failed, error:%d, log:%s",
            XCAM_STR (info.name), error, compile_log.data());
        return NULL;
    }

    SmartPtr<GLShader> shader =
        new GLShader (shader_id, info.type, (info.name ? info.name : "null"));
    return shader;
}

}
