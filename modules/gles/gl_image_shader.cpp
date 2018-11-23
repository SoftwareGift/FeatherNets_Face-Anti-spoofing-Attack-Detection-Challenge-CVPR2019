/*
 * gl_image_shader.cpp - gl image shader implementation
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

#include "gl_image_shader.h"

#define ENABLE_DEBUG_SHADER 0

namespace XCam {

GLImageShader::GLImageShader (const char *name, const SmartPtr<Callback> &cb)
    : Worker (name, cb)
{
    XCAM_OBJ_PROFILING_INIT;
}

GLImageShader::~GLImageShader ()
{
}

XCamReturn
GLImageShader::finish ()
{
    _program->finish ();
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLImageShader::stop ()
{
    return XCAM_RETURN_NO_ERROR;
}

bool
GLImageShader::get_compute_program (SmartPtr<GLComputeProgram> &prog)
{
    prog = _program.dynamic_cast_ptr<GLComputeProgram> ();
    XCAM_FAIL_RETURN (
        ERROR, prog.ptr (), false,
        "GLImageShader(%s) convert to GLComputeProgram failed", XCAM_STR (get_name ()));

    return true;
}

XCamReturn
GLImageShader::work (const SmartPtr<Worker::Arguments> &args)
{
    XCamReturn ret = _program->use ();
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) use program failed", XCAM_STR (get_name ()));

    ret = pre_work (args);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) pre-work failed", XCAM_STR (get_name ()));

#if ENABLE_DEBUG_SHADER
    XCAM_OBJ_PROFILING_START;
#endif

    ret = _program->work ();
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) work failed", XCAM_STR (get_name ()));

#if ENABLE_DEBUG_SHADER
    ret = _program->finish ();
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) finish failed", XCAM_STR (get_name ()));

    char name[XCAM_GL_NAME_LENGTH] = {'\0'};
    snprintf (name, sizeof (name), "%s-%p", XCAM_STR (get_name ()), this);
    XCAM_OBJ_PROFILING_END (name, XCAM_OBJ_DUR_FRAME_NUM);
#endif

    ret = _program->disuse ();
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) disuse program failed", XCAM_STR (get_name ()));

    status_check (args, ret);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLImageShader::pre_work (const SmartPtr<Worker::Arguments> &args)
{
    GLCmdList cmds;

    XCamReturn ret = prepare_arguments (args, cmds);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) prepare arguments failed", XCAM_STR (get_name ()));

    ret = set_commands (cmds);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) set commands failed", XCAM_STR (get_name ()));

    return ret;
}
XCamReturn
GLImageShader::prepare_arguments (const SmartPtr<Worker::Arguments> &args, GLCmdList &cmds)
{
    XCAM_UNUSED (args);
    XCAM_UNUSED (cmds);

    XCAM_LOG_ERROR ("GLImageShader(%s) prepare arguments error", XCAM_STR (get_name ()));
    return XCAM_RETURN_ERROR_GLES;
}

XCamReturn
GLImageShader::set_commands (const GLCmdList &cmds)
{
    GLuint prog_id = _program->get_program_id();
    XCAM_FAIL_RETURN (
        WARNING, prog_id, XCAM_RETURN_ERROR_PARAM,
        "GLImageShader(%s) invalid program id:%d", XCAM_STR (get_name ()), prog_id);

    uint32_t i_count = 0;
    for (GLCmdList::const_iterator iter = cmds.begin (); iter != cmds.end (); ++iter, ++i_count) {
        const SmartPtr<GLCommand> &cmd = *iter;
        XCAM_FAIL_RETURN (
            WARNING, cmd.ptr (), XCAM_RETURN_ERROR_MEM,
            "GLImageShader(%s) command(idx:%d) is NULL", XCAM_STR (get_name ()), i_count);

        XCamReturn ret = cmd->run (prog_id);
        XCAM_FAIL_RETURN (
            WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
            "GLImageShader(%s) command(idx:%d) run failed", XCAM_STR (get_name ()));
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLImageShader::create_compute_program (const GLShaderInfo &info, const char *name)
{
    SmartPtr<GLComputeProgram> program = GLComputeProgram::create_compute_program (name);
    XCAM_FAIL_RETURN (
        ERROR, program.ptr (), XCAM_RETURN_ERROR_GLES,
        "GLImageShader(%s) create compute program(%s) failed", XCAM_STR (get_name ()), XCAM_STR (name));

    XCamReturn ret = program->link_shader (info);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) program(%s) pour shader failed", XCAM_STR (get_name ()), XCAM_STR (name));

    _program = program;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLImageShader::create_compute_program (const GLShaderInfoList &infos, const char *name)
{
    SmartPtr<GLComputeProgram> program = GLComputeProgram::create_compute_program (name);
    XCAM_FAIL_RETURN (
        ERROR, program.ptr (), XCAM_RETURN_ERROR_GLES,
        "GLImageShader(%s) create compute program(%s) failed", XCAM_STR (get_name ()), XCAM_STR (name));

    XCamReturn ret = program->link_shaders (infos);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "GLImageShader(%s) program(%s) pour shaders failed", XCAM_STR (get_name ()), XCAM_STR (name));

    _program = program;

    return XCAM_RETURN_NO_ERROR;
}

};
