/*
 * vk_instance.h - vulkan instance
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

#ifndef XCAM_VK_INSTANCE_H
#define XCAM_VK_INSTANCE_H

#include <vulkan/vulkan_std.h>
#include <xcam_mutex.h>

namespace XCam {

class VKInstance
{
public:
    ~VKInstance ();
    static SmartPtr<VKInstance> get_instance ();

    VkInstance get_id () const {
        return _instance_id;
    }
    VkPhysicalDevice get_physical_dev () const {
        return _physical_device;
    }
    uint32_t get_compute_queue_family_idx () const {
        return _compute_queue_family_idx;
    }
    uint32_t get_graphics_queue_family_idx () const {
        return _graphics_queue_family_idx;
    }
    uint32_t get_mem_type_index (VkMemoryPropertyFlags prop) const;

    SmartPtr<VkAllocationCallbacks> get_allocator () const {
        return _allocator;
    }

private:
    explicit VKInstance (VkInstance id, VkAllocationCallbacks *allocator);
    static SmartPtr<VKInstance> create_instance ();
    XCamReturn query_physical_info ();
    XCamReturn query_queue_info ();

private:
    XCAM_DEAD_COPY (VKInstance);

private:
    static SmartPtr<VKInstance>      _instance;
    static Mutex                     _instance_mutex;

    VkInstance                       _instance_id;
    SmartPtr<VkAllocationCallbacks>  _allocator;
    VkPhysicalDevice                 _physical_device;
    VkPhysicalDeviceProperties       _device_properties;
    VkPhysicalDeviceMemoryProperties _dev_mem_properties;
    uint32_t                         _compute_queue_family_idx;
    uint32_t                         _graphics_queue_family_idx;
};

}

#endif  //XCAM_VK_INSTANCE_H
