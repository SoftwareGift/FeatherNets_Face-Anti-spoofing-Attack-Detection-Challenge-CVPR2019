/*
 * main_dev_manager.cpp - main device manager
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: John Ye <john.ye@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "main_dev_manager.h"

using namespace XCam;

namespace GstXCam {

MainDeviceManager::MainDeviceManager()
{
}

MainDeviceManager::~MainDeviceManager()
{
}

void
MainDeviceManager::handle_message (const SmartPtr<XCamMessage> &msg)
{
    XCAM_UNUSED (msg);
}

void
MainDeviceManager::handle_buffer (const SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (buf.ptr ());
    _ready_buffers.push (buf);
}

SmartPtr<VideoBuffer>
MainDeviceManager::dequeue_buffer ()
{
    SmartPtr<VideoBuffer> ret;
    ret = _ready_buffers.pop (-1);
    return ret;
}

void
MainDeviceManager::pause_dequeue ()
{
    return _ready_buffers.pause_pop ();
}

void
MainDeviceManager::resume_dequeue ()
{
    return _ready_buffers.resume_pop ();
}

};
