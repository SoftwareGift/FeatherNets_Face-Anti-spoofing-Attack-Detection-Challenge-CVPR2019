/*
 * vk_sync.h - Vulkan sync
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

#ifndef XCAM_VK_SYNC_H
#define XCAM_VK_SYNC_H

#include <vulkan/vulkan_std.h>

namespace XCam {

class VKDevice;

class VKFence
{
    friend class VKDevice;
public:
    virtual ~VKFence ();

    XCamReturn reset ();
    XCamReturn wait (uint64_t timeout = UINT64_MAX);

    VkFence get_fence_id () const {
        return _fence_id;
    }

protected:
    explicit VKFence (const SmartPtr<VKDevice> dev, VkFence id);

private:
    XCAM_DEAD_COPY (VKFence);

protected:
    VkFence                  _fence_id;
    SmartPtr<VKDevice>       _dev;
};

}

#endif  //XCAM_VK_SYNC_H
