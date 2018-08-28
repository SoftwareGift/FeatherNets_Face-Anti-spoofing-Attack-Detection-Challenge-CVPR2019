/*
 * vk_pipeline.cpp - Vulkan pipeline
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

#include "vk_pipeline.h"
#include "vulkan_common.h"
#include "vk_cmdbuf.h"

namespace XCam {

VKPipeline::VKPipeline (
    const SmartPtr<VKDevice> dev,
    const ShaderVec &shaders,
    const VKDescriptor::BindingArray &bindings,
    const VKConstRange::VKConstantArray &consts)
    : _pipe_id (VK_NULL_HANDLE)
    , _dev (dev)
    , _shaders (shaders)
    , _bindings (bindings)
    , _push_consts (consts)
{
    _allocator = _dev->get_allocation_cb ();
    xcam_mem_clear (_name);
}

VKPipeline::~VKPipeline ()
{
    if (!_dev.ptr ())
        return;

    VkDevice dev_id = _dev->get_dev_id ();
    if (XCAM_IS_VALID_VK_ID (_pipe_id))
        vkDestroyPipeline (dev_id, _pipe_id, _allocator.ptr ());
}

void
VKPipeline::set_desc_pool (const SmartPtr<VKDescriptor::Pool> pool)
{
    _pool = pool;

    //TODO, check pool status and allocate set, need or not?
}

VkDescriptorSetLayout
VKPipeline::create_desc_set_layout (
    const VKDescriptor::BindingArray &bindings)
{
    VKDescriptor::VkBindingArray array = VKDescriptor::get_vk_layoutbindings (bindings);

    VkDescriptorSetLayoutCreateInfo descriptor_layout = {};
    descriptor_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout.bindingCount = array.size ();
    descriptor_layout.pBindings = array.data ();

    VkDescriptorSetLayout layout = NULL;
    XCAM_VK_CHECK_RETURN (
        ERROR,
        vkCreateDescriptorSetLayout (
            _dev->get_dev_id (), &descriptor_layout, _allocator.ptr (), &layout),
        NULL, "VkPipeline create descriptor set layout failed");

    return layout;
}


VkPipelineLayout
VKPipeline::create_pipeline_layout (
    VkDescriptorSetLayout desc_layout,
    const VKConstRange::VKConstantArray &consts)
{
    VkPipelineLayoutCreateInfo pipe_layout_create_info = {};
    pipe_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipe_layout_create_info.flags = 0;
    pipe_layout_create_info.setLayoutCount = 1;
    pipe_layout_create_info.pSetLayouts = &desc_layout;
    if (!consts.empty()) {
        pipe_layout_create_info.pushConstantRangeCount = consts.size ();
        pipe_layout_create_info.pPushConstantRanges = consts.data ();
    }

    VkPipelineLayout layout = NULL;
    XCAM_VK_CHECK_RETURN (
        ERROR,
        vkCreatePipelineLayout (
            _dev->get_dev_id (), &pipe_layout_create_info, NULL, &layout),
        NULL, "VkPipeline create descriptor set layout failed");

    return layout;
}

SmartPtr<VKPipeline>
VKPipeline::create_compute_pipeline (
    const SmartPtr<VKDevice> dev,
    const SmartPtr<VKShader> shader,
    const VKDescriptor::BindingArray &bindings,
    const VKConstRange::VKConstantArray &consts)
{
    XCAM_FAIL_RETURN (
        ERROR, dev.ptr () && shader.ptr (), NULL,
        "VKDevice create pipeline with error of null device ready.");

    ShaderVec shaders = {shader};
    SmartPtr<VKPipeline> pipe = new VKComputePipeline (dev, shaders, bindings, consts);

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (pipe->ensure_layouts ()), NULL,
        "vk pipeline ensure layouts failed");

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (pipe->ensure_pipeline ()), NULL,
        "vk pipeline ensure pipeline failed");

    return pipe;
}

VKComputePipeline::VKComputePipeline (
    const SmartPtr<VKDevice> dev,
    const ShaderVec &shaders,
    const VKDescriptor::BindingArray &bindings,
    const VKConstRange::VKConstantArray &consts)
    : VKPipeline (dev, shaders, bindings, consts)
    , _pipe_layout (VK_NULL_HANDLE)
    , _desc_layout (VK_NULL_HANDLE)
{
}

VKComputePipeline::~VKComputePipeline ()
{
    if (!_dev.ptr ())
        return;

    VkDevice dev_id = _dev->get_dev_id ();
    if (XCAM_IS_VALID_VK_ID (_pipe_layout))
        vkDestroyPipelineLayout (dev_id, _pipe_layout, _allocator.ptr ());
    if (XCAM_IS_VALID_VK_ID (_desc_layout))
        vkDestroyDescriptorSetLayout (dev_id, _desc_layout, _allocator.ptr ());
}

XCamReturn
VKComputePipeline::ensure_layouts ()
{
    if (!XCAM_IS_VALID_VK_ID (_desc_layout)) {
        _desc_layout = create_desc_set_layout (_bindings);
    }
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID(_desc_layout), XCAM_RETURN_ERROR_VULKAN,
        "vk compute pipeline create desc layout failed");

    if (!XCAM_IS_VALID_VK_ID (_pipe_layout)) {
        _pipe_layout = create_pipeline_layout (_desc_layout, _push_consts);
    }
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID(_pipe_layout), XCAM_RETURN_ERROR_VULKAN,
        "vk compute pipeline create pipeline layout failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKComputePipeline::ensure_pipeline ()
{
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (_desc_layout) && XCAM_IS_VALID_VK_ID (_pipe_layout),
        XCAM_RETURN_ERROR_PARAM,
        "vk compute ensure pipeline failed. need ensure desc_layout and pipe_layout first");

    XCAM_FAIL_RETURN (
        ERROR, !_shaders.empty (), XCAM_RETURN_ERROR_PARAM,
        "vk compute ensure pipeline failed, shader was empty");

    VkComputePipelineCreateInfo pipeline_create_info =
        get_compute_create_info (_shaders[0], _pipe_layout);

    VkPipeline pipe_id;
    XCAM_VK_CHECK_RETURN (
        ERROR, vkCreateComputePipelines (
            _dev->get_dev_id (), 0, 1, &pipeline_create_info, 0, &pipe_id),
        XCAM_RETURN_ERROR_VULKAN, "VK create compute pipeline failed.");

    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (pipe_id));
    _pipe_id = pipe_id;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKComputePipeline::update_bindings (const VKDescriptor::SetBindInfoArray &bind_array)
{
    XCAM_FAIL_RETURN (
        ERROR, _pool.ptr () && XCAM_IS_VALID_VK_ID (_desc_layout), XCAM_RETURN_ERROR_PARAM,
        "vk compute pipeline update bindins failed, pool was not set or desc_layout not ensured");

    if (_desc_set.ptr ())
        _desc_set.release ();

    _desc_set = _pool->allocate_set (bind_array, _desc_layout);
    XCAM_FAIL_RETURN (
        ERROR, _desc_set.ptr (), XCAM_RETURN_ERROR_UNKNOWN,
        "vk compute pipeline update bindins failed to allocate desc_set or update bindings");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKComputePipeline::bind_by (VKCmdBuf &cmd_buf)
{

    VkCommandBuffer buf_id = cmd_buf.get_cmd_buf_id ();
    VkPipeline pipe_id = get_pipeline_id ();
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (buf_id));
    XCAM_FAIL_RETURN (
        ERROR,
        XCAM_IS_VALID_VK_ID (pipe_id) && XCAM_IS_VALID_VK_ID (_pipe_layout) && _desc_set.ptr (),
        XCAM_RETURN_ERROR_PARAM,
        "vk compute pipeline bind command buffer failed, please check pipe_id, pipe_layout and desc_layout.");

    // // bind pipeline sets
    vkCmdBindPipeline (buf_id, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_id);

    // bind descriptor sets
    VkDescriptorSet desc_set_id = _desc_set->get_set_id ();
    vkCmdBindDescriptorSets (
        buf_id, VK_PIPELINE_BIND_POINT_COMPUTE, _pipe_layout,
        0, 1, &desc_set_id, 0, NULL);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKComputePipeline::push_consts_by (
    VKCmdBuf &cmd_buf, const SmartPtr<VKConstRange::VKPushConstArg> &push_const)
{
    VkCommandBuffer cmd_buf_id = cmd_buf.get_cmd_buf_id ();
    XCAM_FAIL_RETURN (
        ERROR,
        XCAM_IS_VALID_VK_ID (cmd_buf_id) && XCAM_IS_VALID_VK_ID (_pipe_layout),
        XCAM_RETURN_ERROR_PARAM,
        "vk compute pipeline push_consts by cmdbuf failed, please check pipe_layout and cmd_buf_id.");

    XCAM_ASSERT (push_const.ptr ());
    VkPushConstantRange const_range;
    xcam_mem_clear (const_range);
    void *ptr = NULL;
    push_const->get_const_data (const_range, ptr);

    XCAM_FAIL_RETURN (
        ERROR,
        const_range.stageFlags == VK_SHADER_STAGE_COMPUTE_BIT,
        XCAM_RETURN_ERROR_PARAM,
        "vk compute pipeline push_consts by cmdbuf failed, please check pipe_layout and cmd_buf_id.");

    vkCmdPushConstants(
        cmd_buf_id, _pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, const_range.offset, const_range.size, ptr);
    return XCAM_RETURN_NO_ERROR;
}

VkComputePipelineCreateInfo
VKComputePipeline::get_compute_create_info (
    const SmartPtr<VKShader> &shader,
    VkPipelineLayout &layout)
{
    VkPipelineShaderStageCreateInfo shader_stage_create_info = {};
    shader_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_create_info.flags = 0;
    shader_stage_create_info.stage = shader->get_shader_stage_flags ();
    shader_stage_create_info.module = shader->get_shader_id ();
    shader_stage_create_info.pName = shader->get_func_name ();

    VkComputePipelineCreateInfo pipeline_create_info = { };
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.pNext = NULL;
    pipeline_create_info.flags = 0;
    pipeline_create_info.stage = shader_stage_create_info;
    pipeline_create_info.layout = layout;

    return pipeline_create_info;
}

}
