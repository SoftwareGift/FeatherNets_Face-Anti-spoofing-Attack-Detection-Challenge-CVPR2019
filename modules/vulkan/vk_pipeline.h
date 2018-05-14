/*
 * vk_pipeline.h - Vulkan pipeline
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

#ifndef XCAM_VK_PIPELINE_H
#define XCAM_VK_PIPELINE_H

#include <vulkan/vulkan_std.h>
#include <vulkan/vk_descriptor.h>
#include <vulkan/vk_device.h>
#include <vulkan/vk_shader.h>

namespace XCam {

class VKPipeline
{
public:
    static SmartPtr<VKPipeline>
    create_compute_pipeline (
        const SmartPtr<VKDevice> dev,
        const SmartPtr<VKShader> shader,
        const VKDescriptor::BindingArray &bindings,
        const VKConstRange::VKConstantArray &consts);

    virtual ~VKPipeline ();

    VkPipeline get_pipeline_id () const {
        return _pipe_id;
    }
    const char *get_name () const {
        return _name;
    }
    void set_desc_pool (const SmartPtr<VKDescriptor::Pool> pool);
    //interface
    virtual XCamReturn update_bindings (const VKDescriptor::SetBindInfoArray &bind_array) = 0;

    // inter-functions, called by VKCmdBuf
    virtual XCamReturn bind_by (VKCmdBuf &cmd_buf) = 0;
    virtual XCamReturn push_consts_by (
        VKCmdBuf &cmd_buf, const SmartPtr<VKConstRange::VKPushConstArg> &push_const) = 0;

protected:
    explicit VKPipeline (
        const SmartPtr<VKDevice> dev,
        const ShaderVec &shaders,
        const VKDescriptor::BindingArray &bindings,
        const VKConstRange::VKConstantArray &consts);

    VkDescriptorSetLayout create_desc_set_layout (
        const VKDescriptor::BindingArray &bindings);

    VkPipelineLayout create_pipeline_layout (
        VkDescriptorSetLayout desc_layout,
        const VKConstRange::VKConstantArray &consts);

    // intra virtual functions
    virtual XCamReturn ensure_layouts () = 0;
    virtual XCamReturn ensure_pipeline () = 0;

private:
    XCAM_DEAD_COPY (VKPipeline);

protected:
    VkPipeline                       _pipe_id;
    char                             _name [XCAM_VK_NAME_LENGTH];

    SmartPtr<VKDevice>               _dev;
    SmartPtr<VkAllocationCallbacks>  _allocator;
    ShaderVec                        _shaders;
    VKDescriptor::BindingArray       _bindings;
    VKConstRange::VKConstantArray    _push_consts;
    SmartPtr<VKDescriptor::Pool>     _pool;
};

class VKComputePipeline
    : public VKPipeline
{
    friend class VKPipeline;

public:
    static VkComputePipelineCreateInfo
    get_compute_create_info (const SmartPtr<VKShader> &shader, VkPipelineLayout &layout);

    ~VKComputePipeline ();

    //inherit from VKPipeline
    XCamReturn update_bindings (const VKDescriptor::SetBindInfoArray &bind_array);

protected:
    explicit VKComputePipeline (
        const SmartPtr<VKDevice> dev,
        const ShaderVec &shaders,
        const VKDescriptor::BindingArray &bindings,
        const VKConstRange::VKConstantArray &consts);

    //virtual functions from VKPipeline
    XCamReturn ensure_layouts ();
    XCamReturn ensure_pipeline ();
    XCamReturn bind_by (VKCmdBuf &cmd_buf);
    XCamReturn push_consts_by (
        VKCmdBuf &cmd_buf, const SmartPtr<VKConstRange::VKPushConstArg> &push_const);

private:
    VkPipelineLayout                 _pipe_layout;
    VkDescriptorSetLayout            _desc_layout;
    SmartPtr<VKDescriptor::Set>      _desc_set;
};

}

#endif  //XCAM_VK_PIPELINE_H
