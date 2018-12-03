/*
 * vk_descriptor.h - Vulkan descriptor
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

#ifndef XCAM_VK_DESCRIPTOR_H
#define XCAM_VK_DESCRIPTOR_H

#include <vulkan/vulkan_std.h>
#include <vulkan/vk_memory.h>
#include <map>

namespace XCam {

class VKDevice;

namespace VKDescriptor {

class SetLayoutBinding
{
public:
    virtual ~SetLayoutBinding ();
    const VkDescriptorSetLayoutBinding &get_vk_binding () const {
        return _binding;
    }
    VkDescriptorType get_desc_type () const {
        return _binding.descriptorType;
    }
    uint32_t get_index () const {
        return _binding.binding;
    }

protected:
    explicit SetLayoutBinding (
        VkDescriptorType type, VkShaderStageFlags stage, uint32_t idx, uint32_t count);

private:
    XCAM_DEAD_COPY (SetLayoutBinding);
protected:
    VkDescriptorSetLayoutBinding    _binding;
};

template <VkShaderStageFlags stage>
class LayoutBinding
    : public SetLayoutBinding
{
public:
    explicit LayoutBinding (VkDescriptorType type, uint32_t idx)
        : SetLayoutBinding (type, stage, idx, 1)
    {}
};

typedef LayoutBinding<VK_SHADER_STAGE_COMPUTE_BIT> ComputeLayoutBinding;
typedef LayoutBinding<VK_SHADER_STAGE_VERTEX_BIT> VetexLayoutBinding;
typedef LayoutBinding<VK_SHADER_STAGE_FRAGMENT_BIT> FragmentLayoutBinding;

typedef std::vector<SmartPtr<SetLayoutBinding>>  BindingArray;

typedef std::vector<VkDescriptorSetLayoutBinding>  VkBindingArray;

VkBindingArray get_vk_layoutbindings (const BindingArray &array);

struct SetBindInfo {
    SmartPtr<SetLayoutBinding>   layout;
    VKBufDesc                    desc;
};

typedef std::vector<SetBindInfo> SetBindInfoArray;

class Pool;
class Set {
public:
    explicit Set (VkDescriptorSet set_id, const SmartPtr<Pool> pool);
    ~Set ();
    XCamReturn update_set (const SetBindInfoArray &bind_array);
    VkDescriptorSet get_set_id () const {
        return _set_id;
    }

private:
    XCAM_DEAD_COPY (Set);

    VkDescriptorSet           _set_id;
    SetBindInfoArray          _bind_array;
    SmartPtr<Pool>            _pool;
};

class Pool
    : public RefObj
{
    friend class Set;
public:
    explicit Pool (const SmartPtr<VKDevice> dev);
    ~Pool ();
    bool add_set_bindings (const BindingArray &binds);
    XCamReturn create ();
    const SmartPtr<VKDevice> &get_device() const {
        return _dev;
    }

    SmartPtr<Set> allocate_set (
        const SetBindInfoArray &bind_array, VkDescriptorSetLayout layout);

private:
    XCAM_DEAD_COPY (Pool);
    void add_binding (const SmartPtr<SetLayoutBinding> &bind);
    void destroy_desc_set (VkDescriptorSet set_id);

private:
    typedef std::map<VkDescriptorType, uint32_t>  TypeTable;

    VkDescriptorPool                _pool_id;
    uint32_t                        _set_size;
    TypeTable                       _types;
    const SmartPtr<VKDevice>        _dev;
};

}

namespace VKConstRange {

class VKPushConstArg {
public:
    virtual ~VKPushConstArg () {}
    virtual bool get_const_data (VkPushConstantRange &range, void *& ptr) = 0;
};

typedef std::vector<SmartPtr<VKPushConstArg>> VKPushConstArgs;

typedef std::vector<VkPushConstantRange> VKConstantArray;

template <VkShaderStageFlags stage>
VkPushConstantRange
get_constants (uint32_t size, uint32_t offset)
{
    VkPushConstantRange range = {};
    range.stageFlags = stage;
    range.offset = offset;
    range.size = size;
    return range;
}

#define get_compute_consts get_constants<VK_SHADER_STAGE_COMPUTE_BIT>
}

}

#endif  //XCAM_VK_DESCRIPTOR_H
