/*
 * gl_program.cpp - GL Program
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

#include "gl_program.h"
#include "gl_shader.h"

namespace XCam {

GLenum get_gl_error ()
{
    return glGetError();
}

GLProgram::GLProgram (GLuint id, const char *name)
    : _program_id (id)
    , _state (GLProgram::StateIntiated)
{
    strncpy (_name, name, sizeof (_name) - 1);
}

GLProgram::~GLProgram ()
{
    disuse ();
    clear_shaders ();
    if (_program_id)
        glDeleteProgram (_program_id);
}

SmartPtr<GLProgram>
GLProgram::create_program (const char *name)
{
    GLuint program_id = glCreateProgram();
    XCAM_FAIL_RETURN (
        ERROR, program_id, NULL,
        "Create GL program(%s) failed, err_no:%d",
        get_gl_error (), XCAM_STR (name));

    if (!program_id)
        return NULL;

    return new GLProgram (program_id, name ? name : "null");
}

XCamReturn
GLProgram::attach_shader (const SmartPtr<GLShader> &shader)
{
    GLuint shader_id = shader->get_shader_id ();
    XCAM_ASSERT (shader_id);
    XCAM_FAIL_RETURN (
        ERROR, _shaders.find (shader_id) == _shaders.end (),
        XCAM_RETURN_ERROR_PARAM,
        "GL program(:%s)already have shader (id:%d), do not attach twice",
        get_name(), shader_id);

    glAttachShader(_program_id, shader_id);
    XCAM_FAIL_RETURN (
        ERROR, glGetError () == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GL program(:%s)attach shader (id:%d) failed, error_no:%d",
        get_name(), shader_id, glGetError ());

    _shaders.insert (ShaderList::value_type (shader_id, shader));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLProgram::detach_shader (const SmartPtr<GLShader> &shader)
{
    GLuint shader_id = shader->get_shader_id ();
    XCAM_ASSERT (shader_id);
    ShaderList::iterator pos = _shaders.find (shader_id);

    XCAM_FAIL_RETURN (
        WARNING, pos == _shaders.end (),
        XCAM_RETURN_ERROR_PARAM,
        "GL program(:%s) do not need to detach shader (id:%d) which is not exist",
        get_name(), shader_id);

    glDetachShader (_program_id, shader_id);
    if (glGetError () != GL_NO_ERROR) {
        XCAM_LOG_WARNING (
            "GL program(:%s)detach shader (id:%d) failed but continued, error_no:%d",
            get_name(), shader_id, glGetError ());
    }
    _shaders.erase (pos);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLProgram::clear_shaders ()
{
    for (ShaderList::iterator i = _shaders.begin ();
            i != _shaders.end (); ++i) {
        SmartPtr<GLShader> shader = i->second;
        glDetachShader (_program_id, shader->get_shader_id ());
    }
    _shaders.clear ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLProgram::link ()
{
    XCAM_ASSERT (_program_id);
    glLinkProgram (_program_id);

    GLint status;
    std::vector<char> link_log;
    glGetProgramiv(_program_id, GL_LINK_STATUS, &status);
    if(status == GL_FALSE) {
        GLint length;
        glGetProgramiv(_program_id, GL_INFO_LOG_LENGTH, &length);
        link_log.resize (length + 1);
        glGetProgramInfoLog(_program_id, length, &length, &link_log[0]);
        XCAM_LOG_ERROR(
            "GL program(:%s)link failed, error_no:%d, link log:%s",
            get_name(), glGetError (), link_log.data());
        return XCAM_RETURN_ERROR_GLES;
    }

    _state = StateLinked;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLProgram::use ()
{
    XCAM_ASSERT (_program_id);
    XCAM_FAIL_RETURN (
        WARNING, _state == StateLinked && !_shaders.empty (),
        XCAM_RETURN_ERROR_PARAM,
        "GL program(:%s) use must be called after link", get_name());

    glUseProgram (_program_id);

    XCAM_FAIL_RETURN (
        ERROR, glGetError () == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GL program(:%s) use failed, error_no:%d",
        get_name(), glGetError ());

    _state = StateInUse;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLProgram::disuse ()
{
    if (_state != StateInUse)
        return XCAM_RETURN_BYPASS;

    glUseProgram (0);
    _state = StateLinked;
    return XCAM_RETURN_NO_ERROR;
}

}
