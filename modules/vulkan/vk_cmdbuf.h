/*
 * vk_cmdbuf.h - Vulkan command buffer
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

#ifndef XCAM_VK_CMD_BUF_H
#define XCAM_VK_CMD_BUF_H

#include <vulkan/vulkan_std.h>
#include <vulkan/vk_descriptor.h>
#include <vulkan/vk_device.h>

namespace XCam {

class VKCmdBuf
{
public:
    struct GroupSize {
        uint32_t x;
        uint32_t y;
        uint32_t z;
        GroupSize (uint32_t x_, uint32_t y_, uint32_t z_)
            : x (x_)
            , y (y_)
            , z (z_)
        {}
    };

    class DispatchParam {
        friend class VKCmdBuf;
    public:
        DispatchParam (const SmartPtr<VKPipeline> &p, uint32_t x, uint32_t y = 1, uint32_t z = 1);
        virtual ~DispatchParam ();
        bool update_push_consts (VKConstRange::VKPushConstArgs & push_consts);
    protected:
        virtual XCamReturn fill_cmd_buf (VKCmdBuf &buf);
        XCAM_DEAD_COPY (DispatchParam);

    protected:
        GroupSize                       _group_size;
        const SmartPtr<VKPipeline>      _pipeline;
        VKConstRange::VKPushConstArgs   _push_consts;
    };

    class Pool
        : public RefObj
    {
        friend class VKCmdBuf;
    public:
        ~Pool ();
        SmartPtr<VKCmdBuf> allocate_buffer ();
        void free_buffer (VkCommandBuffer buf_id);

    private:
        explicit Pool (const SmartPtr<VKDevice> dev, VkCommandPool id);
        XCAM_DEAD_COPY (Pool);
    private:
        VkCommandPool         _pool_id;
        SmartPtr<VKDevice>    _dev;
    };

public:
    static SmartPtr<VKCmdBuf>
    create_command_buffer (const SmartPtr<VKDevice> dev, const SmartPtr<Pool> pool = NULL);
    static SmartPtr<Pool> create_pool (const SmartPtr<VKDevice> dev, VkFlags queue_flag);
    virtual ~VKCmdBuf ();

    VkCommandBuffer get_cmd_buf_id () const {
        return _cmd_buf_id;
    }

    XCamReturn record (const SmartPtr<DispatchParam> param);

    // for fill_cmd_buf
    XCamReturn dispatch (const GroupSize &group);

protected:
    explicit VKCmdBuf (const SmartPtr<Pool> pool, VkCommandBuffer buf_id);

private:
    XCAM_DEAD_COPY (VKCmdBuf);

protected:
    VkCommandBuffer                  _cmd_buf_id;

    SmartPtr<Pool>                   _pool;
};

}

#endif  //XCAM_VK_CMD_BUF_H
