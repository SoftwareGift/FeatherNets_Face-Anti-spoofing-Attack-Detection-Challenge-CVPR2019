/*
 * vk_cmdbuf.cpp - Vulkan command buffer
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

#include "vk_cmdbuf.h"
#include "vulkan_common.h"
#include "vk_pipeline.h"

namespace XCam {

VKCmdBuf::Pool::Pool (const SmartPtr<VKDevice> dev, VkCommandPool id)
    : _pool_id (id)
    , _dev (dev)
{
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (id));
}

VKCmdBuf::Pool::~Pool ()
{
    if (XCAM_IS_VALID_VK_ID (_pool_id))
        _dev->destroy_cmd_pool (_pool_id);
}

SmartPtr<VKCmdBuf>
VKCmdBuf::Pool::allocate_buffer ()
{
    //VkCommandBufferAllocateInfo info {};
    VkCommandBuffer cmd_buf_id = _dev->allocate_cmd_buffer (_pool_id);
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (cmd_buf_id), NULL,
        "VKCmdBuf allocate cmd buffer failed");
    return new VKCmdBuf (this, cmd_buf_id);
}

void
VKCmdBuf::Pool::free_buffer (VkCommandBuffer buf_id)
{
    XCAM_ASSERT (_dev.ptr ());
    _dev->free_cmd_buffer (_pool_id, buf_id);
}

SmartPtr<VKCmdBuf::Pool>
VKCmdBuf::create_pool (const SmartPtr<VKDevice> dev, VkFlags queue_flag)
{
    VkCommandPool pool_id = dev->create_cmd_pool (queue_flag);
    XCAM_FAIL_RETURN (
        ERROR, XCAM_IS_VALID_VK_ID (pool_id), NULL,
        "VKCmdBuf create_pool failed");
    SmartPtr<VKCmdBuf::Pool> pool = new VKCmdBuf::Pool (dev, pool_id);
    return pool;
}

SmartPtr<VKCmdBuf>
VKCmdBuf::create_command_buffer (
    const SmartPtr<VKDevice> dev,
    const SmartPtr<VKCmdBuf::Pool> pool)
{
    XCAM_FAIL_RETURN (
        ERROR, dev.ptr (), NULL,
        "VKCmdBuf create command buffer failed");

    SmartPtr<VKCmdBuf::Pool> cmdbuf_pool = pool;
    if (!pool.ptr()) {
        cmdbuf_pool = create_pool (dev, VK_QUEUE_COMPUTE_BIT);
    }

    XCAM_FAIL_RETURN (
        ERROR, cmdbuf_pool.ptr (), NULL,
        "VKCmdBuf create command pool failed");

    return cmdbuf_pool->allocate_buffer ();
}

VKCmdBuf::VKCmdBuf (const SmartPtr<VKCmdBuf::Pool> pool, VkCommandBuffer buf_id)
    : _cmd_buf_id (buf_id)
    , _pool (pool)
{
}

VKCmdBuf::~VKCmdBuf ()
{
    if (_pool.ptr () && XCAM_IS_VALID_VK_ID (_cmd_buf_id))
        _pool->free_buffer (_cmd_buf_id);
}

VKCmdBuf::DispatchParam::DispatchParam (const SmartPtr<VKPipeline> &p, uint32_t x, uint32_t y, uint32_t z)
    : _group_size (x, y, z)
    , _pipeline (p)
{
}

VKCmdBuf::DispatchParam::~DispatchParam ()
{
}

bool
VKCmdBuf::DispatchParam::update_push_consts (VKConstRange::VKPushConstArgs & push_consts)
{
    _push_consts = push_consts;
    return true;
}

XCamReturn
VKCmdBuf::DispatchParam::fill_cmd_buf (VKCmdBuf &buf)
{
    XCamReturn ret = _pipeline->bind_by (buf);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret),
        ret, "VKCmdBuf DispatchParam fill command buffer failed when binding pipeline");

    for (size_t i = 0; i < _push_consts.size (); ++i) {
        XCAM_RETURN_CHECK (
            ERROR, _pipeline->push_consts_by (buf, _push_consts[i]),
            "VKCmdBuf DispatchParam fill command buffer failed when push consts (:%d)", i);
    }
    return buf.dispatch (_group_size);
}

XCamReturn
VKCmdBuf::record (const SmartPtr<DispatchParam> param)
{
    VkCommandBufferBeginInfo buf_begin_info = {};
    buf_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    buf_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    buf_begin_info.pInheritanceInfo = NULL;

    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_cmd_buf_id));
    XCAM_VK_CHECK_RETURN (
        ERROR, vkBeginCommandBuffer (_cmd_buf_id, &buf_begin_info),
        XCAM_RETURN_ERROR_VULKAN, "VKCmdBuf begin command buffer failed");

    XCamReturn ret = param->fill_cmd_buf (*this);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret),
        ret, "VKCmdBuf dispatch params failed");

    XCAM_VK_CHECK_RETURN (
        ERROR, vkEndCommandBuffer (_cmd_buf_id),
        XCAM_RETURN_ERROR_VULKAN, "VKCmdBuf begin command buffer failed");

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKCmdBuf::dispatch (const GroupSize &group)
{
    XCAM_FAIL_RETURN (
        ERROR, group.x * group.y * group.z > 0,
        XCAM_RETURN_ERROR_VULKAN, "VKCmdBuf dispatch params failed");

    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_cmd_buf_id));
    vkCmdDispatch (_cmd_buf_id, group.x, group.y, group.z);
    return XCAM_RETURN_NO_ERROR;
}

}
