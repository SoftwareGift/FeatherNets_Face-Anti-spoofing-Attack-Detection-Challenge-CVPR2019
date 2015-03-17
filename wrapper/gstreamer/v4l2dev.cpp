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
#include "atomisp_device.h"

namespace XCam {

SmartPtr<V4l2Device> V4l2Dev::_device (NULL);
Mutex V4l2Dev::_mutex;
const char* V4l2Dev::_device_name (NULL);

SmartPtr<V4l2Device>
V4l2Dev::instance ()
{
    SmartLock lock(_mutex);
    if (_device.ptr())
        return _device;
    _device = new AtomispDevice (_device_name);
    return _device;
}

};
