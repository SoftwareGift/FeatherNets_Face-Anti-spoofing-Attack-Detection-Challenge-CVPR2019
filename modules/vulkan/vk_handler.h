/*
 * vk_handler.h - vulkan image handler class
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

#ifndef XCAM_VK_HANDLER_H
#define XCAM_VK_HANDLER_H

#include <vulkan/vulkan_std.h>
#include <image_handler.h>

namespace XCam {

class VKDevice;

class VKHandler
    : public ImageHandler
{
public:
    explicit VKHandler (const SmartPtr<VKDevice> &dev, const char* name = "vk-handler");
    ~VKHandler ();
    const SmartPtr<VKDevice> &get_vk_device () const {
        return _device;
    }

    // derive from ImageHandler
    virtual XCamReturn finish ();
    virtual XCamReturn terminate ();

protected:
    SmartPtr<BufferPool> create_allocator ();

private:
    XCAM_DEAD_COPY (VKHandler);

protected:
    SmartPtr<VKDevice>      _device;
};

}

#endif //XCAM_VK_HANDLER_H
