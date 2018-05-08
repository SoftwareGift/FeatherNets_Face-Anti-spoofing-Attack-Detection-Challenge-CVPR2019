/*
 * gl_compute_program.cpp - GL compute program implementation
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

#include "gl_compute_program.h"

namespace XCam {

GLGroupsSize GLComputeProgram::_max_groups_size;

GLComputeProgram::GLComputeProgram (GLuint id, const char *name)
    : GLProgram (id, name)
{
    if (_max_groups_size.x == 0 && _max_groups_size.y == 0 && _max_groups_size.z == 0) {
        get_max_groups_size (_max_groups_size);
    }
}

GLComputeProgram::~GLComputeProgram ()
{
}

SmartPtr<GLComputeProgram>
GLComputeProgram::create_compute_program (const char *name)
{
    GLuint prog_id = glCreateProgram ();
    XCAM_FAIL_RETURN (
        ERROR, prog_id, NULL,
        "create GL program(%s) failed, error_no:%d", XCAM_STR (name), glGetError ());

    SmartPtr<GLComputeProgram> compute_prog = new GLComputeProgram (prog_id, name);
    XCAM_FAIL_RETURN (
        ERROR, compute_prog.ptr (), NULL,
        "create GL compute program(%s) failed", XCAM_STR (name));

    return compute_prog;
}

bool query_max_groups_size (GLuint idx, GLint &value)
{
    glGetIntegeri_v (GL_MAX_COMPUTE_WORK_GROUP_COUNT, idx, &value);

    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, false,
        "GLComputeProgram query max groups size failed, idx:%d error_no:%d", idx, error);

    return true;
}

bool
GLComputeProgram::get_max_groups_size (GLGroupsSize &size)
{
    XCAM_FAIL_RETURN (
        ERROR,
        query_max_groups_size (0, size.x) &&
        query_max_groups_size (1, size.y) &&
        query_max_groups_size (2, size.z),
        false,
        "GLComputeProgram(%s) get max groups size failed", get_name ());

    return true;
}

bool
GLComputeProgram::check_groups_size (const GLGroupsSize &size)
{
    XCAM_FAIL_RETURN (
        ERROR,
        size.x > 0 && size.x <= _max_groups_size.x &&
        size.y > 0 && size.y <= _max_groups_size.y &&
        size.z > 0 && size.z <= _max_groups_size.z,
        false,
        "GLComputeProgram(%s) invalid groups size: %dx%dx%d",
        get_name (), size.x, size.y, size.z);

    return true;
}

bool
GLComputeProgram::set_groups_size (const GLGroupsSize &size)
{
    XCAM_FAIL_RETURN (
        ERROR, check_groups_size (size), false,
        "GLComputeProgram(%s) set groups size failed, groups size: %dx%dx%d",
        get_name (), size.x, size.y, size.z);

    _groups_size = size;

    return true;
}

XCamReturn
GLComputeProgram::dispatch ()
{
    XCAM_FAIL_RETURN (
        ERROR, check_groups_size (_groups_size), XCAM_RETURN_ERROR_PARAM,
        "GLComputeProgram(%s) dispatch invalid groups size: %dx%dx%d",
        get_name (), _groups_size.x, _groups_size.y, _groups_size.z);

    glDispatchCompute (_groups_size.x, _groups_size.y, _groups_size.z);

    GLenum error = glGetError ();
    XCAM_FAIL_RETURN (
        ERROR, error == GL_NO_ERROR, XCAM_RETURN_ERROR_GLES,
        "GLComputeProgram(%s) dispatch failed, error_no:%d, groups size: %dx%dx%d",
        get_name (), error, _groups_size.x, _groups_size.y, _groups_size.z);

    return XCAM_RETURN_NO_ERROR;
}

}
