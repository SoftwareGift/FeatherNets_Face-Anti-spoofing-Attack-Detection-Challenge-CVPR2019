/*
 * gl_image_shader.h - gl image shader class
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

#ifndef XCAM_GL_IMAGE_SHADER_H
#define XCAM_GL_IMAGE_SHADER_H

#include <worker.h>
#include <image_handler.h>
#include <gles/gles_std.h>
#include <gles/gl_command.h>
#include <gles/gl_compute_program.h>

namespace XCam {

struct GLArgs
    : Worker::Arguments
{
private:
    SmartPtr<ImageHandler::Parameters> _param;

public:
    explicit GLArgs (const SmartPtr<ImageHandler::Parameters> &param = NULL) : _param (param) {}
    inline const SmartPtr<ImageHandler::Parameters> &get_param () const {
        return _param;
    }
    inline void set_param (const SmartPtr<ImageHandler::Parameters> &param) {
        XCAM_ASSERT (param.ptr ());
        _param = param;
    }
};

class GLImageShader
    : public Worker
{
public:
    explicit GLImageShader (const char *name, const SmartPtr<Callback> &cb = NULL);
    virtual ~GLImageShader ();

    XCamReturn set_commands (const GLCmdList &cmds);
    bool get_compute_program (SmartPtr<GLComputeProgram> &prog);

    // derived from Worker
    virtual XCamReturn work (const SmartPtr<Arguments> &args);
    virtual XCamReturn finish ();
    virtual XCamReturn stop ();

    XCamReturn create_compute_program (const GLShaderInfo &info, const char *name = NULL);
    XCamReturn create_compute_program (const GLShaderInfoList &infos, const char *name  = NULL);

private:
    XCamReturn pre_work (const SmartPtr<Worker::Arguments> &args);
    virtual XCamReturn prepare_arguments (const SmartPtr<Worker::Arguments> &args, GLCmdList &cmds);

private:
    XCAM_DEAD_COPY (GLImageShader);

private:
    SmartPtr<GLProgram>        _program;

    XCAM_OBJ_PROFILING_DEFINES;
};

}
#endif // XCAM_GL_IMAGE_SHADER_H
