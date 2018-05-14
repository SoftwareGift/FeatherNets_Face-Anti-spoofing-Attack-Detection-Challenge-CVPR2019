/*
 * vk_shader.cpp - vulkan shader module
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

#include "vk_shader.h"
#include "vk_device.h"
#include "file_handle.h"

namespace XCam {

VKShader::VKShader (SmartPtr<VKDevice> dev, VkShaderModule id, const char *name)
    : _device (dev)
    , _shader_id (id)
    , _shader_stage (VK_SHADER_STAGE_COMPUTE_BIT)
{
    XCAM_IS_VALID_VK_ID (id);
    xcam_mem_clear (_name);
    if (name)
        strncpy (_name, name, XCAM_VK_NAME_LENGTH - 1);
    strncpy (_func_name, "main", XCAM_VK_NAME_LENGTH - 1);
}

VKShader::~VKShader ()
{
    if (XCAM_IS_VALID_VK_ID (_shader_id))
        _device->destroy_shader_id (_shader_id);
}

void
VKShader::set_func_name (const char *name)
{
    XCAM_ASSERT (name);
    strncpy (_func_name, name, XCAM_VK_NAME_LENGTH - 1);
}

void
VKShader::set_name (const char *name)
{
    XCAM_ASSERT (name);
    strncpy (_name, name, XCAM_VK_NAME_LENGTH - 1);
}


}
