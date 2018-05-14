/*
 * vulkan_std.h - vulkan common
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

#ifndef XCAM_VK_STD_H
#define XCAM_VK_STD_H

#include <xcam_std.h>
#include <vulkan/vulkan.h>

#define XCAM_VK_CHECK_RETURN(LEVEL, vk_exp, failed_value, format, ...) \
    do {                                                               \
        VkResult err = vk_exp;                                         \
        XCAM_FAIL_RETURN (LEVEL, err == VK_SUCCESS, failed_value,      \
                          format ", vk_error(%d:%s)", ## __VA_ARGS__,  \
                          (int)err, vk_error_str(err));                \
    } while (0)


#define XCAM_VK_NAME_LENGTH 256

#define XCAM_IS_VALID_VK_ID(id) (VK_NULL_HANDLE != (id))

#define XCAM_VK_SHADER_PATH "XCAM_VK_SHADER_PATH"
#define XCAM_DEFAULT_VK_SHADER_PATH xcam_default_shader_path()

#endif
