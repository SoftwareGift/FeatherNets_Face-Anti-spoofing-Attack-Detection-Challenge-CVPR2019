/*
 * vk_stitcher.h - Vulkan stitcher class
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

#ifndef XCAM_VK_STITCHER_H
#define XCAM_VK_STITCHER_H

#include <interface/stitcher.h>
#include <vulkan/vulkan_std.h>
#include <vulkan/vk_handler.h>

namespace XCam {

namespace VKSitcherPriv {
class StitcherImpl;
class CbGeoMap;
};

class VKStitcher
    : public VKHandler
    , public Stitcher
{
    friend class VKSitcherPriv::StitcherImpl;
    friend class VKSitcherPriv::CbGeoMap;

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
    explicit VKStitcher (const SmartPtr<VKDevice> &dev, const char *name = "VKStitcher");
    ~VKStitcher ();

    // derived from VKHandler
    virtual XCamReturn terminate ();

protected:
    // interface derive from Stitcher
    XCamReturn stitch_buffers (const VideoBufferList &in_bufs, SmartPtr<VideoBuffer> &out_buf);

    // derived from VKHandler
    XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    void geomap_done (
        const SmartPtr<ImageHandler> &handler,
        const SmartPtr<ImageHandler::Parameters> &param, const XCamReturn error);

private:
    SmartPtr<VKSitcherPriv::StitcherImpl>    _impl;
};

}
#endif // XCAM_VK_STITCHER_H
