/*
 * vk_geomap_handler.h - vulkan geometry map handler class
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

#ifndef XCAM_VK_GEOMAP_HANDLER_H
#define XCAM_VK_GEOMAP_HANDLER_H

#include <xcam_utils.h>
#include <interface/geo_mapper.h>
#include <vulkan/vulkan_std.h>
#include <vulkan/vk_worker.h>
#include <vulkan/vk_handler.h>

namespace XCam {

class VKGeoMapHandler
    : public VKHandler, public GeoMapper
{
public:
    struct PushConstsProp {
        uint     in_img_width;
        uint     in_img_height;
        uint     out_img_width;
        uint     out_img_height;
        uint     lut_width;
        uint     lut_height;
        float    lut_step[4];
        float    lut_std_step[2];

        PushConstsProp ();
    };

public:
    explicit VKGeoMapHandler (const SmartPtr<VKDevice> dev, const char* name = "vk-geomap-handler");

    bool set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height);

    XCamReturn remap (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf);
    void geomap_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

private:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    virtual bool init_factors ();

private:
    SmartPtr<VKWorker>               _worker;
    PushConstsProp                   _image_prop;
    VKDescriptor::BindingArray       _binding_layout;

    SmartPtr<VKBuffer>               _lut_buf;
    uint32_t                         _lut_width;
    uint32_t                         _lut_height;
};

}
#endif // XCAM_VK_GEOMAP_HANDLER_H
