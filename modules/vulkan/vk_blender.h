/*
 * vk_blender.h - vulkan blender class
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

#ifndef XCAM_VK_BLENDER_H
#define XCAM_VK_BLENDER_H

#include <vulkan/vulkan_std.h>
#include <vulkan/vk_handler.h>
#include <interface/blender.h>

#define VK_BLENDER_ALIGN_X 8
#define VK_BLENDER_ALIGN_Y 4

#define XCAM_VK_MAX_LEVEL 4
#define XCAM_VK_DEFAULT_LEVEL 2

namespace XCam {

namespace VKBlenderPriv {
class BlenderImpl;
}

class VKBlender
    : public VKHandler, public Blender
{
    friend class VKBlenderPriv::BlenderImpl;
    friend SmartPtr<VKHandler> create_vk_blender (const SmartPtr<VKDevice> &dev);

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

    class Sync
    {
    public:
         Sync (uint32_t threshold)
            : _count (0), _threshold (threshold)
        {}

        void increment () {
            ++_count;
        }
        void reset () {
            _count = 0;
        }
        bool is_synced () {
            return (_threshold == _count);
        }

    private:
        uint32_t          _count;
        const uint32_t    _threshold;
    };

    enum BufIdx {
        BufIdx0    = 0,
        BufIdx1,
        BufIdxMax
    };

public:
    ~VKBlender ();

    void gauss_scale_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);
    void lap_trans_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);
    void blend_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);
    void reconstruct_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error);

protected:
    explicit VKBlender (const SmartPtr<VKDevice> dev, const char *name = "VKBlender");

    // derived from Blender interface
    XCamReturn blend (
        const SmartPtr<VideoBuffer> &in0, const SmartPtr<VideoBuffer> &in1, SmartPtr<VideoBuffer> &out);

    // derived from VKHandler
    XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    XCamReturn start_work (const SmartPtr<Parameters> &param);

    XCamReturn set_output_info (const SmartPtr<ImageHandler::Parameters> &param);

private:
    SmartPtr<VKBlenderPriv::BlenderImpl>    _impl;
};

extern SmartPtr<VKHandler> create_vk_blender (const SmartPtr<VKDevice> &dev);

}

#endif // XCAM_VK_BLENDER_H
