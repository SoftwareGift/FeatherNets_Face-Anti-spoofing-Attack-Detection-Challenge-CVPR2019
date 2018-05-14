/*
 * vk_shader.h - Vulkan shader module
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

#ifndef XCAM_VK_SHADER_H
#define XCAM_VK_SHADER_H

#include <vulkan/vulkan_std.h>

namespace XCam {

class VKDevice;

class VKShader
{
    friend class VKDevice;
public:
    ~VKShader ();

    VkShaderModule get_shader_id () const {
        return _shader_id;
    }
    VkShaderStageFlagBits get_shader_stage_flags () const {
        return _shader_stage;
    }
    void set_func_name (const char *name);
    void set_name (const char *name);
    const char *get_func_name () const {
        return _func_name;
    }
    const char *get_name () const {
        return _name;
    }

private:
    explicit VKShader (SmartPtr<VKDevice> dev, VkShaderModule id, const char *name = "null");

private:
    XCAM_DEAD_COPY (VKShader);

private:
    //static ShaderTable               _shader_cache;
    //static Mutex                     _cache_mutex;
    SmartPtr<VKDevice>               _device;
    VkShaderModule                   _shader_id;
    VkShaderStageFlagBits            _shader_stage;
    char                             _func_name [XCAM_VK_NAME_LENGTH];
    char                             _name [XCAM_VK_NAME_LENGTH];
};

typedef std::vector<SmartPtr<VKShader>> ShaderVec;

}

#endif  //XCAM_VK_SHADER_H
