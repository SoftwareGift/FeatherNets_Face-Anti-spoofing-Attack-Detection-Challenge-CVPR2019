/*
 * gl_copy_handler.h - gl copy handler class
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

#ifndef XCAM_GL_COPY_HANDER_H
#define XCAM_GL_COPY_HANDER_H

#include <xcam_utils.h>
#include <gles/gl_image_shader.h>
#include <gles/gl_image_handler.h>

namespace XCam {

class GLCopyShader
    : public GLImageShader
{
public:
    struct Args : GLArgs {
        uint32_t                  index;
        Rect                      in_area, out_area;
        SmartPtr<GLBuffer>        in_buf, out_buf;

        Args (const SmartPtr<ImageHandler::Parameters> &param)
            : GLArgs (param)
            , index (0)
        {}
    };

public:
    explicit GLCopyShader (const SmartPtr<Worker::Callback> &cb)
        : GLImageShader ("GLCopyShader", cb)
    {}

    ~GLCopyShader () {}

private:
    virtual XCamReturn prepare_arguments (const SmartPtr<Worker::Arguments> &args, GLCmdList &cmds);
};

class GLCopyHandler
    : public GLImageHandler
{
    friend class CbCopyShader;

public:
    GLCopyHandler (const char *name = "GLCopyHandler");
    ~GLCopyHandler ();

    bool set_copy_area (uint32_t idx, const Rect &in_area, const Rect &out_area);
    XCamReturn copy (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf);

protected:
    //derived from ImageHandler
    virtual XCamReturn terminate ();

    //derived from GLImageHandler
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    SmartPtr<GLCopyShader> create_copy_shader ();
    XCamReturn start_copy_shader (const SmartPtr<ImageHandler::Parameters> &param);
    void copy_shader_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

private:
    uint32_t                      _index;
    Rect                          _in_area;
    Rect                          _out_area;
    SmartPtr<GLCopyShader>        _copy_shader;
};

}
#endif // XCAM_GL_COPY_HANDER_H
