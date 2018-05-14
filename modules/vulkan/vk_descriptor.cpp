/*
 * vk_descriptor.cpp - Vulkan descriptor
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

#include "vk_descriptor.h"
#include "vk_device.h"

namespace XCam {

namespace VKDescriptor {

SetLayoutBinding::SetLayoutBinding (
    VkDescriptorType type, VkShaderStageFlags stage, uint32_t idx, uint32_t count)
{
    xcam_mem_clear (_binding);
    _binding.binding = idx;
    _binding.descriptorType = type;
    _binding.descriptorCount = count;
    _binding.stageFlags = stage;
    _binding.pImmutableSamplers = NULL;
}

SetLayoutBinding::~SetLayoutBinding ()
{
}

VkBindingArray
get_vk_layoutbindings (const BindingArray &array)
{
    VkBindingArray ret;
    ret.reserve (array.size ());
    for (size_t i = 0; i < array.size (); ++i) {
        ret.push_back(array[i]->get_vk_binding ());
    }
    return ret;
}

Pool::Pool (const SmartPtr<VKDevice> dev)
    : _pool_id (VK_NULL_HANDLE)
    , _set_size (0)
    , _dev (dev)
{}

Pool::~Pool ()
{
    if (XCAM_IS_VALID_VK_ID (_pool_id)) {
        _dev->destroy_desc_pool (_pool_id);
    }
}

void
Pool::add_binding (const SmartPtr<SetLayoutBinding> &bind)
{
    VkDescriptorSetLayoutBinding vk_binding = bind->get_vk_binding ();
    Pool::TypeTable::iterator i = _types.find (vk_binding.descriptorType);
    if (i == _types.end ())
        _types.insert (i, Pool::TypeTable::value_type (vk_binding.descriptorType, vk_binding.descriptorCount));
    else
        i->second += vk_binding.descriptorCount;
}

bool
Pool::add_set_bindings (const BindingArray &binds)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_IS_VALID_VK_ID (_pool_id), false,
        "vk desriptor pool was inited, cannot add new binding.");

    for (BindingArray::const_iterator i = binds.begin (); i != binds.end (); ++i) {
        add_binding (*i);
    }
    ++_set_size;

    return true;
}

XCamReturn
Pool::create ()
{
    XCAM_FAIL_RETURN (
        ERROR, !_types.empty (), XCAM_RETURN_ERROR_PARAM,
        "vk desriptor pool cannot create since no types added.");

    XCAM_FAIL_RETURN (
        ERROR, _dev.ptr (), XCAM_RETURN_ERROR_PARAM,
        "vk desriptor pool cannot create, device is null");

    std::vector<VkDescriptorPoolSize> pool_sizes;
    pool_sizes.reserve (_types.size ());
    for (Pool::TypeTable::iterator i = _types.begin (); i != _types.end(); ++i) {
        VkDescriptorPoolSize new_size = {};
        new_size.type = i->first;
        new_size.descriptorCount = i->second;
        pool_sizes.push_back (new_size);
    }

    XCAM_ASSERT (_set_size);
    VkDescriptorPoolCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    create_info.maxSets = _set_size;
    create_info.poolSizeCount = pool_sizes.size ();
    create_info.pPoolSizes = pool_sizes.data ();

    _pool_id = _dev->create_desc_pool (create_info);

    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (_pool_id), XCAM_RETURN_ERROR_VULKAN,
        "vk desriptor pool create pool_id failed");

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<Set>
Pool::allocate_set (const SetBindInfoArray &bind_array, VkDescriptorSetLayout layout)
{
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (_pool_id), NULL,
        "vk desriptor pool allocate set failed, pool was not ready");

#if 0
    XCAM_FAIL_RETURN (
        ERROR, bind_array.size () == bufs.size (), NULL,
        "vk desriptor pool allocate set failed, bindings and bufs sizes are not matched");
#endif

    XCAM_FAIL_RETURN (
        ERROR, _set_size > 0, NULL,
        "vk desriptor pool allocate set failed, bindings and bufs sizes are not matched");

    //TODO remove binds types from _types

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = _pool_id;
    alloc_info.pSetLayouts = &layout;
    alloc_info.descriptorSetCount = 1;

    VkDescriptorSet desc_set_id = _dev->allocate_desc_set (alloc_info);
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (desc_set_id) > 0, NULL,
        "vk desriptor pool allocate set failed");
    SmartPtr<Set> new_set = new Set (desc_set_id, this);

    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (new_set->update_set (bind_array)), NULL,
        "vk descriptor pool update set failed");

    --_set_size;
    return new_set;
}

void
Pool::destroy_desc_set (VkDescriptorSet set_id)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_pool_id));
    _dev->free_desc_set (set_id, _pool_id);
}

Set::Set (VkDescriptorSet set_id, const SmartPtr<Pool> pool)
    : _set_id (set_id)
    , _pool (pool)
{
}

Set::~Set ()
{
    if (XCAM_IS_VALID_VK_ID (_set_id)) {
        _pool->destroy_desc_set (_set_id);
    }
}

XCamReturn
Set::update_set (const SetBindInfoArray &bind_array)
{
    std::vector<VkWriteDescriptorSet> write_desc_info (bind_array.size ());
    for (uint32_t i = 0; i < bind_array.size (); ++i) {
        const SetBindInfo &bind_info = bind_array[i];
        SmartPtr<SetLayoutBinding> bind = bind_info.layout;
        XCAM_ASSERT (bind.ptr () && bind_info.desc.buf.ptr ());

        VkWriteDescriptorSet &info = write_desc_info[i];
        xcam_mem_clear (info);
        info.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        info.dstSet = _set_id;
        info.dstBinding = bind->get_index ();
        info.descriptorCount = 1;
        info.descriptorType = bind->get_desc_type();
        info.pBufferInfo = &bind_info.desc.desc_info;
    }
    SmartPtr<VKDevice> dev = _pool->get_device ();
    XCAM_ASSERT (dev.ptr ());
    XCamReturn ret = dev->update_desc_set (write_desc_info);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "vk descriptor pool update set failed");

    _bind_array = bind_array;
    return ret;
}

}

}
