/*
 * gl_compute_program.h - GL compute program class
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

#ifndef XCAM_GL_COMPUTE_PROGRAM_H
#define XCAM_GL_COMPUTE_PROGRAM_H

#include <gles/gl_program.h>

namespace XCam {

struct GLGroupsSize {
    GLint x, y, z;
    GLGroupsSize () : x (0), y (0), z (0) {}
};

class GLComputeProgram
    : public GLProgram
{
public:
    static SmartPtr<GLComputeProgram> create_compute_program (const char *name = NULL);
    ~GLComputeProgram ();

    bool set_groups_size (const GLGroupsSize &size);
    XCamReturn dispatch ();

private:
    explicit GLComputeProgram (GLuint id, const char *name);

    bool get_max_groups_size (GLGroupsSize &size);
    bool check_groups_size (const GLGroupsSize &size);

private:
    XCAM_DEAD_COPY (GLComputeProgram);

private:
    GLGroupsSize               _groups_size;
    static GLGroupsSize        _max_groups_size;
};

}

#endif // XCAM_GL_COMPUTE_PROGRAM_H
