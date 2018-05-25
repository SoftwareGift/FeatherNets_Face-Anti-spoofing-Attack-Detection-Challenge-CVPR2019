/*
 * gl_program.h - GL Program
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

#ifndef XCAM_GL_PROGRAM_H
#define XCAM_GL_PROGRAM_H

#include <gles/gles_std.h>
#include <map>

namespace XCam {

class GLShader;

class GLProgram
{
public:
    typedef std::map<GLuint, SmartPtr<GLShader>> ShaderList;
    enum State {
        StateIntiated  = 0,
        StateLinked    = 2,
        StateInUse     = 3,
    };

    virtual ~GLProgram ();
    static SmartPtr<GLProgram> create_program (const char *name = NULL);
    GLuint get_program_id () const {
        return _program_id;
    }
    const char *get_name () {
        return _name;
    }
    XCamReturn attach_shader (const SmartPtr<GLShader> &shader);
    XCamReturn detach_shader (const SmartPtr<GLShader> &shader);
    XCamReturn clear_shaders ();

    XCamReturn link ();
    XCamReturn use ();
    XCamReturn disuse ();

    virtual XCamReturn work ();
    virtual XCamReturn finish ();

protected:
    explicit GLProgram (GLuint id, const char *name);

private:
    XCAM_DEAD_COPY (GLProgram);

private:
    ShaderList    _shaders;
    GLuint        _program_id;
    State         _state;
    char          _name [XCAM_GL_NAME_LENGTH];
};

}

#endif  //XCAM_GL_PROGRAM_H
