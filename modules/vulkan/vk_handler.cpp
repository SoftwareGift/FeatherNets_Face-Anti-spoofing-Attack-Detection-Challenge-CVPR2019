/*
 * vk_handler.cpp - vulkan image handler class
 *
 *  Copyright (c) 2017 Intel Corporation
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

#include "vk_handler.h"
#include "vk_device.h"
#include "vk_video_buf_allocator.h"

namespace XCam {

VKHandler::VKHandler (const SmartPtr<VKDevice> &dev, const char* name)
    : ImageHandler (name)
    , _device (dev)
{
}

VKHandler::~VKHandler ()
{
}


XCamReturn
VKHandler::finish ()
{
    if (_device.ptr ())
        _device->compute_queue_wait_idle ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
VKHandler::terminate ()
{
    finish ();
    return ImageHandler::terminate ();
}

SmartPtr<BufferPool>
VKHandler::create_allocator ()
{
    return new VKVideoBufAllocator (_device);
}

}
