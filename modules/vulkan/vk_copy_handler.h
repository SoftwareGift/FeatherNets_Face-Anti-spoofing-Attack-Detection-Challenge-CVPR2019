/*
 * vk_copy_handler.h - vulkan copy handler
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

#ifndef XCAM_VK_COPY_HANDLER_H
#define XCAM_VK_COPY_HANDLER_H

#include <xcam_utils.h>
#include <vulkan/vulkan_std.h>
#include <vulkan/vk_worker.h>
#include <vulkan/vk_handler.h>
#include <vulkan/vk_descriptor.h>

namespace XCam {

#if 0
class VKCopyArguments
    : public VKWorker::VKArguments
{
public:
    explicit VKCopyArguments (const SmartPtr<VKBuffer> in, SmartPtr<VKBuffer> out);

private:
    virtual XCamReturn prepare_bindings (VKDescriptor::SetBindInfoArray &binding_array);
private:
    SmartPtr<VKBuffer>        _in_buf;
    SmartPtr<VKBuffer>        _out_buf;
};
#endif

class VKCopyHandler
    : public VKHandler
{
public:
    struct PushConstsProp {
        uint    in_img_width;
        uint    in_x_offset;
        uint    out_img_width;
        uint    out_x_offset;
        uint    copy_width;

        PushConstsProp ();
    };

public:
    explicit VKCopyHandler (const SmartPtr<VKDevice> &dev, const char* name = "vk-copy-handler");

    bool set_copy_area (uint32_t idx, const Rect &in_area, const Rect &out_area);
    uint32_t get_index () {
        return _index;
    }

    XCamReturn copy (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf);
    void copy_done (
        const SmartPtr<Worker> &worker,
        const SmartPtr<Worker::Arguments> &base,
        const XCamReturn error);

private:
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    SmartPtr<VKWorker>               _worker;
    PushConstsProp                   _image_prop;
    VKDescriptor::BindingArray       _binding_layout;

    uint32_t                         _index;
    Rect                             _in_area;
    Rect                             _out_area;
};

}
#endif //XCAM_VK_COPY_HANDLER_H
