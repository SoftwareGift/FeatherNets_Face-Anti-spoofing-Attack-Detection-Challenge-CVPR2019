/*
 * v4l2dev.cpp - wrapper of V4l2Device
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
 */

#include "v4l2dev.h"
namespace XCam {

SmartPtr<MainDeviceManager> DeviceManagerInstance::_device_manager (NULL);
Mutex               DeviceManagerInstance::_device_manager_mutex;

SmartPtr<MainDeviceManager>&
DeviceManagerInstance::device_manager_instance ()
{
    SmartLock lock (_device_manager_mutex);
    if (_device_manager.ptr())
        return _device_manager;

    _device_manager = new MainDeviceManager;
    return _device_manager;
}

MainDeviceManager::MainDeviceManager()
{
}

MainDeviceManager::~MainDeviceManager()
{
}

void
MainDeviceManager::handle_message (SmartPtr<XCamMessage> &msg)
{
    XCAM_UNUSED (msg);
}

void
MainDeviceManager::handle_buffer (SmartPtr<VideoBuffer> &buf)
{
    pthread_mutex_lock (&bufs_mutex);
    bufs.push (buf);
    if (bufs.size() == 1)
        pthread_cond_signal (&bufs_cond);
    pthread_mutex_unlock (&bufs_mutex);
}

};
