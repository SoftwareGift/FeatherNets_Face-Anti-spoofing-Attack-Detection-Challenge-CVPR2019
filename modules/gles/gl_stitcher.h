/*
 * gl_stitcher.h - GL stitcher class
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

#ifndef XCAM_GL_STITCHER_H
#define XCAM_GL_STITCHER_H

#include <interface/stitcher.h>
#include <gles/gles_std.h>
#include <gles/gl_image_handler.h>

namespace XCam {

namespace GLSitcherPriv {
class StitcherImpl;
class CbGeoMap;
class CbBlender;
class CbCopier;
};

class GLStitcher
    : public GLImageHandler
    , public Stitcher
{
    friend class GLSitcherPriv::StitcherImpl;
    friend class GLSitcherPriv::CbGeoMap;
    friend class GLSitcherPriv::CbBlender;
    friend class GLSitcherPriv::CbCopier;

public:
    struct StitcherParam
        : ImageHandler::Parameters
    {
        uint32_t in_buf_num;
        SmartPtr<VideoBuffer> in_bufs[XCAM_STITCH_MAX_CAMERAS];

        StitcherParam ()
            : Parameters (NULL, NULL)
            , in_buf_num (0)
        {}
    };

public:
    explicit GLStitcher (const char *name = "GLStitcher");
    ~GLStitcher ();

    // derived from GLImageHandler
    virtual XCamReturn terminate ();

protected:
    // interface derive from Stitcher
    XCamReturn stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf);

    // derived from GLImageHandler
    XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    void dewarp_done (
        const SmartPtr<ImageHandler> &handler,
        const SmartPtr<ImageHandler::Parameters> &param, const XCamReturn error);
    void blender_done (
        const SmartPtr<ImageHandler> &handler,
        const SmartPtr<ImageHandler::Parameters> &param, const XCamReturn error);
    void copier_done (
        const SmartPtr<ImageHandler> &handler,
        const SmartPtr<ImageHandler::Parameters> &param, const XCamReturn error);

private:
    SmartPtr<GLSitcherPriv::StitcherImpl>    _impl;
};

}
#endif // XCAM_GL_STITCHER_H