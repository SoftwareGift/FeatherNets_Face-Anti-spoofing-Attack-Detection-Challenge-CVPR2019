/*
 * vk_instance.cpp - vulkan instance
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

#include "vk_instance.h"
#include "vulkan_common.h"

#define APP_NAME "xcam"
#define ENGINE_NAME "xcam"

#define XCAM_INVALID_VK_QUEUE_IDX UINT32_MAX

namespace XCam {

extern void vk_init_error_string ();

SmartPtr<VKInstance> VKInstance::_instance;
Mutex VKInstance::_instance_mutex;

VKInstance::~VKInstance ()
{
    if (XCAM_IS_VALID_VK_ID (_instance_id))
        vkDestroyInstance (_instance_id, _allocator.ptr ());
}

VKInstance::VKInstance (VkInstance id, VkAllocationCallbacks *allocator)
    : _instance_id (id)
    , _allocator (allocator)
    , _physical_device (NULL)
    , _compute_queue_family_idx (XCAM_INVALID_VK_QUEUE_IDX)
    , _graphics_queue_family_idx (XCAM_INVALID_VK_QUEUE_IDX)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (id));
    xcam_mem_clear (_device_properties);
    xcam_mem_clear (_dev_mem_properties);
}

SmartPtr<VKInstance>
VKInstance::get_instance ()
{
    SmartLock locker (_instance_mutex);
    if (!_instance.ptr ()) {
        vk_init_error_string ();
        _instance = create_instance ();
    }
    return _instance;
}

SmartPtr<VKInstance>
VKInstance::create_instance ()
{
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = APP_NAME;
    app_info.applicationVersion = 0;
    app_info.pEngineName = ENGINE_NAME;
    app_info.engineVersion = xcam_version ();
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstance id;
    VkInstanceCreateInfo inst_create_info = {};
    inst_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_create_info.pApplicationInfo = &app_info;
    inst_create_info.enabledExtensionCount = 0; // TODO, add extensions
    XCAM_VK_CHECK_RETURN(
        ERROR, vkCreateInstance (&inst_create_info, NULL, &id),
        NULL, "create vk instance failed");

    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (id));
    SmartPtr<VKInstance> vk_instance = new VKInstance (id, NULL);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (vk_instance->query_physical_info ()), NULL,
        "vk instance query physical info failed");

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (vk_instance->query_queue_info ()), NULL,
        "vk instance query queue info failed");

    return vk_instance;
}

static const char *s_device_types[] = {
    "OTHER_DEVICE",
    "INTEGRATED_GPU",
    "DISCRETE_GPU",
    "VIRTUAL_GPU",
    "CPU_TYPE",
};

static const char*
device_type_to_str(VkPhysicalDeviceType type)
{
    size_t number = sizeof (s_device_types) / sizeof (s_device_types[0]);
    assert (number == 5);
    if ((size_t)type < number)
        return s_device_types [type];
    return "UNKNOWN_TYPE";
}

XCamReturn
VKInstance::query_physical_info ()
{
#define MAX_DEV_NUM 256
    VkPhysicalDevice devs[MAX_DEV_NUM];
    uint32_t dev_num = 0;
    XCAM_VK_CHECK_RETURN (
        ERROR, vkEnumeratePhysicalDevices (_instance_id, &dev_num, NULL),
        XCAM_RETURN_ERROR_VULKAN, "enum vk physical devices failed");
    XCAM_FAIL_RETURN (
        ERROR, dev_num, XCAM_RETURN_ERROR_VULKAN,
        "There is NO vk physical devices");

    dev_num = XCAM_MIN (dev_num, MAX_DEV_NUM);
    vkEnumeratePhysicalDevices (_instance_id, &dev_num, devs);

    VkPhysicalDevice gpu_dev[VK_PHYSICAL_DEVICE_TYPE_RANGE_SIZE] = {};

    VkPhysicalDeviceProperties dev_prop;
    for (uint32_t i = 0; i < dev_num; ++i) {
        vkGetPhysicalDeviceProperties (devs[i], &dev_prop);

        if (dev_prop.deviceType < VK_PHYSICAL_DEVICE_TYPE_BEGIN_RANGE ||
                dev_prop.deviceType > VK_PHYSICAL_DEVICE_TYPE_END_RANGE) {
            continue;
        }
        if (gpu_dev[dev_prop.deviceType]) {
            XCAM_LOG_WARNING (
                "double vk physical dev, type:%d, name:%s",
                dev_prop.deviceType, dev_prop.deviceName);
            continue;
        }
        gpu_dev[dev_prop.deviceType] = devs[i];
#if 0
        printf ("found vk physical dev_id:%d, name:%s, type:%d, API:%d\n",
                dev_prop.deviceID, dev_prop.deviceName,
                device_type_to_str (dev_prop.deviceType), dev_prop.apiVersion);
#endif
    }

    if (gpu_dev[VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU])
        _physical_device = gpu_dev[VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU];
    else if (gpu_dev[VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU])
        _physical_device = gpu_dev[VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU];
    else if (gpu_dev[VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU])
        _physical_device = gpu_dev[VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU];
    else if (gpu_dev[VK_PHYSICAL_DEVICE_TYPE_CPU]) {
        _physical_device = gpu_dev[VK_PHYSICAL_DEVICE_TYPE_CPU];
        XCAM_LOG_WARNING ("vk device select physical CPU, performance may slow down");
    } else {
        XCAM_LOG_ERROR ("did NOT find available vk physical device");
        return XCAM_RETURN_ERROR_VULKAN;
    }

    vkGetPhysicalDeviceProperties (_physical_device, &dev_prop);
    XCAM_LOG_INFO ("choose vk physical dev properties dev_id:%d, name:%s, type:%s, API:%d\n",
                   dev_prop.deviceID, dev_prop.deviceName,
                   device_type_to_str (dev_prop.deviceType), dev_prop.apiVersion);
    _device_properties = dev_prop;
    vkGetPhysicalDeviceMemoryProperties (_physical_device, &_dev_mem_properties);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKInstance::query_queue_info ()
{
    XCAM_ASSERT (_physical_device);
    // get queue family porperties
    uint32_t queue_count = 0;
#define MAX_QUEUE_FAMILY_NUM 256
    VkQueueFamilyProperties queue_family[MAX_QUEUE_FAMILY_NUM];

    vkGetPhysicalDeviceQueueFamilyProperties (
        _physical_device, &queue_count, NULL);
    XCAM_FAIL_RETURN (
        ERROR, queue_count, XCAM_RETURN_ERROR_VULKAN,
        "There is NO vk physical devices");

    if (queue_count > MAX_QUEUE_FAMILY_NUM)
        queue_count = MAX_QUEUE_FAMILY_NUM;

    vkGetPhysicalDeviceQueueFamilyProperties (
        _physical_device, &queue_count, queue_family);

    for (uint32_t i = 0; i < queue_count; ++i) {
        if (queue_family[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            _compute_queue_family_idx = i;
        }
        if (queue_family[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            _graphics_queue_family_idx = i;
        }
    }

    XCAM_FAIL_RETURN (
        ERROR,
        _compute_queue_family_idx != XCAM_INVALID_VK_QUEUE_IDX &&
        _graphics_queue_family_idx != XCAM_INVALID_VK_QUEUE_IDX,
        XCAM_RETURN_ERROR_VULKAN,
        "There is NO vk compute/graphics queue family");

    return XCAM_RETURN_NO_ERROR;
}

uint32_t
VKInstance::get_mem_type_index (VkMemoryPropertyFlags prop) const
{
    for (uint32_t i = 0; i < _dev_mem_properties.memoryTypeCount; ++i) {
        if (((uint32_t)(_dev_mem_properties.memoryTypes[i].propertyFlags) & prop) == prop)
            return i;
    }
    return (uint32_t)(-1);
}

}
