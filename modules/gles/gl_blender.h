/*
 * gl_blender.h - gl blender class
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_GL_BLENDER_H
#define XCAM_GL_BLENDER_H

#include <interface/blender.h>
#include <gles/gl_image_handler.h>

#define XCAM_GL_PYRAMID_MAX_LEVEL 4
#define XCAM_GL_PYRAMID_DEFAULT_LEVEL 3

namespace XCam {

namespace GLBlenderPriv {
class BlenderPrivConfig;
};

class GLBlender
    : public GLImageHandler, public Blender
{
    friend class GLBlenderPriv::BlenderPrivConfig;
    friend SmartPtr<GLImageHandler> create_gl_blender ();

public:
    struct BlenderParam : ImageHandler::Parameters {
        SmartPtr<VideoBuffer> in1_buf;

        BlenderParam (
            const SmartPtr<VideoBuffer> &in0,
            const SmartPtr<VideoBuffer> &in1,
            const SmartPtr<VideoBuffer> &out)
            : Parameters (in0, out)
            , in1_buf (in1)
        {}
    };

    enum BufIdx {
        Idx0 = 0,
        Idx1,
        BufIdxCount
    };

public:
    ~GLBlender ();

    //derived from GLHandler
    virtual XCamReturn finish ();
    virtual XCamReturn terminate ();

    void gauss_scale_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);
    void lap_trans_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);
    void blend_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);
    void reconstruct_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);

protected:
    explicit GLBlender (const char *name = "GLBlender");

    //derived from Blender interface
    XCamReturn blend (
        const SmartPtr<VideoBuffer> &in0,
        const SmartPtr<VideoBuffer> &in1,
        SmartPtr<VideoBuffer> &out_buf);

    //derived from SoftHandler
    XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    SmartPtr<GLBlenderPriv::BlenderPrivConfig>    _priv_config;
};

extern SmartPtr<GLImageHandler> create_gl_blender ();
}

#endif // XCAM_GL_BLENDER_H
