/*
 * soft_stitcher.h - soft stitcher class
 *
 *  Copyright (c) 2017 Intel Corporation
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

#ifndef XCAM_SOFT_STITCHER_H
#define XCAM_SOFT_STITCHER_H

#include <xcam_std.h>
#include <interface/stitcher.h>
#include <soft/soft_handler.h>

namespace XCam {

namespace SoftSitcherPriv {
class StitcherImpl;
class CbGeoMap;
class CbBlender;
class CbCopyTask;
};

class SoftStitcher
    : public SoftHandler
    , public Stitcher
{
    friend class SoftSitcherPriv::StitcherImpl;
    friend class SoftSitcherPriv::CbGeoMap;
    friend class SoftSitcherPriv::CbBlender;
    friend class SoftSitcherPriv::CbCopyTask;

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
    explicit SoftStitcher (const char *name = "SoftStitcher");
    ~SoftStitcher ();

    //derived from SoftHandler
    virtual XCamReturn terminate ();

protected:
    // interface derive from Stitcher
    XCamReturn stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf);

    //derived from SoftHandler
    XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    // handler done, call back functions
    XCamReturn start_task_count (
        const SmartPtr<SoftStitcher::StitcherParam> &param);
    void dewarp_done (
        const SmartPtr<ImageHandler> &handler,
        const SmartPtr<ImageHandler::Parameters> &param, const XCamReturn error);
    void blender_done (
        const SmartPtr<ImageHandler> &handler,
        const SmartPtr<ImageHandler::Parameters> &param, const XCamReturn error);
    void copy_task_done (
        const SmartPtr<Worker> &worker,
        const SmartPtr<Worker::Arguments> &base, const XCamReturn error);

private:
    SmartPtr<SoftSitcherPriv::StitcherImpl> _impl;
};

}

#endif //XCAM_SOFT_STITCHER_H
