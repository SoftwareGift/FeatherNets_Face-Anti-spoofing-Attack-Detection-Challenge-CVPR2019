/*
 * vulkan_common.cpp - vulkan common
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

#include "vulkan_common.h"
#include <map>

#define VK_STR_INSERT(ERR)    \
    vk_errors.insert (VkErrorMap::value_type(VK_ ##ERR, #ERR))

namespace XCam {

typedef std::map <uint32_t, const char*> VkErrorMap;

static VkErrorMap vk_errors;

void vk_init_error_string ()
{
    if (!vk_errors.empty ())
        return;

    VK_STR_INSERT (SUCCESS);
    VK_STR_INSERT (NOT_READY);
    VK_STR_INSERT (TIMEOUT);
    VK_STR_INSERT (EVENT_SET);
    VK_STR_INSERT (EVENT_RESET);
    VK_STR_INSERT (INCOMPLETE);
    VK_STR_INSERT (ERROR_OUT_OF_HOST_MEMORY);
    VK_STR_INSERT (ERROR_OUT_OF_DEVICE_MEMORY);
    VK_STR_INSERT (ERROR_INITIALIZATION_FAILED);
    VK_STR_INSERT (ERROR_DEVICE_LOST);
    VK_STR_INSERT (ERROR_MEMORY_MAP_FAILED);
    VK_STR_INSERT (ERROR_LAYER_NOT_PRESENT);
    VK_STR_INSERT (ERROR_FEATURE_NOT_PRESENT);
    VK_STR_INSERT (ERROR_INCOMPATIBLE_DRIVER);
    VK_STR_INSERT (ERROR_TOO_MANY_OBJECTS);
    VK_STR_INSERT (ERROR_FORMAT_NOT_SUPPORTED);
    VK_STR_INSERT (ERROR_FRAGMENTED_POOL);
}

const char*
vk_error_str(VkResult id)
{
    VkErrorMap::iterator i = vk_errors.find (id);
    if (i == vk_errors.end ())
        return "VkUnKnown";
    return i->second;
}

const std::string
xcam_default_shader_path ()
{
    std::string home = "~";
    const char *env = std::getenv ("HOME");
    if (env)
        home.assign (env, strlen (env));

    return home + "/.xcam/vk";
}

}
