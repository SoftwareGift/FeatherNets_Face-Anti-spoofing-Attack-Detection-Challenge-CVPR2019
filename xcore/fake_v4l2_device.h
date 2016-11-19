/*
 * fake_v4l2_device.h - fake v4l2 device
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
 * Author: Jia Meng <jia.meng@intel.com>
 */

#ifndef XCAM_FAKE_V4L2_DEVICE_H
#define XCAM_FAKE_V4L2_DEVICE_H

#include "v4l2_device.h"

namespace XCam {

class FakeV4l2Device
    : public V4l2Device
{
public:
    FakeV4l2Device ()
        : V4l2Device ("/dev/null")
    {}

    int io_control (int cmd, void *arg)
    {
        XCAM_UNUSED (arg);

        int ret = 0;
        switch (cmd) {
        case VIDIOC_ENUM_FMT:
            ret = -1;
            break;
        default:
            break;
        }
        return ret;
    }
};

};
#endif // XCAM_FAKE_V4L2_DEVICE_H
