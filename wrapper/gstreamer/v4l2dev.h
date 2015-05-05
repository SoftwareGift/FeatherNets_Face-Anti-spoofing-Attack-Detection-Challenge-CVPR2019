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
#include "device_manager.h"
#include "v4l2dev.h"
#include "atomisp_device.h"
#include "device_manager.h"
#include "isp_controller.h"
#include "isp_image_processor.h"
#if HAVE_LIBCL
#include "cl_3a_image_processor.h"
#endif
#if HAVE_IA_AIQ
#include "x3a_analyzer_aiq.h"
#endif
#include "x3a_analyzer_simple.h"


#include <queue>
#include <unistd.h>
#include <pthread.h>

namespace XCam {

class DeviceManagerInstance;
class MainDeviceManager;

class DeviceManagerInstance {
public:
    static SmartPtr<MainDeviceManager>&  device_manager_instance();

private:
    DeviceManagerInstance ();
    static SmartPtr<MainDeviceManager>  _device_manager;
    static Mutex            _device_manager_mutex;
};

class MainDeviceManager
    : public DeviceManager
{
public:
    MainDeviceManager ();
    ~MainDeviceManager ();

    SmartPtr<V4l2Device>& get_capture_device () {
        return _device;
    }

    SmartPtr<V4l2SubDevice>& get_event_device () {
        return _subdevice;
    }

    SmartPtr<IspController>& get_isp_controller () {
        return _isp_controller;
    }

    SmartPtr<X3aAnalyzer>& get_analyzer () {
        return _3a_analyzer;
    }

protected:
    virtual void handle_message (SmartPtr<XCamMessage> &msg);
    virtual void handle_buffer (SmartPtr<VideoBuffer> &buf);

public:
    std::queue< SmartPtr<VideoBuffer> > bufs;
    pthread_mutex_t         bufs_mutex;
    pthread_cond_t          bufs_cond;
    std::queue< SmartPtr<VideoBuffer> > release_bufs;
    pthread_mutex_t         release_mutex;

#if HAVE_LIBCL
public:
    void set_cl_image_processor (SmartPtr<CL3aImageProcessor> &processor) {
        _cl_image_processor = processor;
    }

    SmartPtr<CL3aImageProcessor> &get_cl_image_processor () {
        return _cl_image_processor;
    }

private:
    SmartPtr<CL3aImageProcessor> _cl_image_processor;
#endif
};

};

#endif  //__V4L2DEV_H__
