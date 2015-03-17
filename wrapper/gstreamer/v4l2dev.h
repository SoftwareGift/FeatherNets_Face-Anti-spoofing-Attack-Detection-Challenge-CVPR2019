/*
 * v4l2dev.h - wrapper of V4l2Device
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

#ifndef __V4L2DEV_H__
#define __V4L2DEV_H__

#include <stdint.h>
#include "xcam_defs.h"
#include "xcam_mutex.h"
#include "v4l2_buffer_proxy.h"
#include "v4l2_device.h"

namespace XCam {

class V4l2Dev {
public:
    static SmartPtr<V4l2Device> instance();
    static const char*      _device_name;

private:
    V4l2Dev ();
    static SmartPtr<V4l2Device> _device;
    static Mutex        _mutex;
};

};

#endif  //__V4L2DEV_H__
