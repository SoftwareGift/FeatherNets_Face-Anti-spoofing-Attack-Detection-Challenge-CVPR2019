/*
 * vk_sync.cpp - Vulkan sync
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

#include "vk_sync.h"
#include "vk_device.h"

namespace XCam {

VKFence::~VKFence ()
{
    if (_dev.ptr () && XCAM_IS_VALID_VK_ID (_fence_id))
        _dev->destroy_fence (_fence_id);
}

VKFence::VKFence (const SmartPtr<VKDevice> dev, VkFence id)
    : _fence_id (id)
    , _dev (dev)
{
}

XCamReturn
VKFence::reset ()
{
    XCAM_ASSERT (_dev.ptr ());
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_fence_id));

    return _dev->reset_fence (_fence_id);
}

XCamReturn
VKFence::wait (uint64_t timeout)
{
    XCAM_ASSERT (_dev.ptr ());
    XCAM_ASSERT (XCAM_IS_VALID_VK_ID (_fence_id));
    return _dev->wait_for_fence (_fence_id, timeout);
}

}
