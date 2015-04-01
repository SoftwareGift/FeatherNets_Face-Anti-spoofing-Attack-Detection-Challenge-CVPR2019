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

    SmartPtr<V4l2Device> get_device() {
        return _device;
    }
    SmartPtr<V4l2SubDevice> get_sub_device() {
        return _sub_device;
    }

    static void set_capture_device_name (const char*);
    static void set_event_device_name (const char*);
    static void set_cpf_file_name (const char*);

    SmartPtr<X3aAnalyzer>& get_x3a_analyzer ();

protected:
    virtual void handle_message (SmartPtr<XCamMessage> &msg);
    virtual void handle_buffer (SmartPtr<VideoBuffer> &buf);

public:
    std::queue< SmartPtr<VideoBuffer> > bufs;
    pthread_mutex_t         bufs_mutex;
    pthread_cond_t          bufs_cond;
    std::queue< SmartPtr<VideoBuffer> > release_bufs;
    pthread_mutex_t         release_mutex;

private:
    SmartPtr<V4l2Device>        _device;
    SmartPtr<V4l2SubDevice>     _sub_device;
    SmartPtr<IspController>     _isp_controller;
    SmartPtr<X3aAnalyzer>       _x3a_analyzer;
    SmartPtr<ImageProcessor>        _image_processor;

    static const char*          _capture_device_name;
    static const char*          _event_device_name;
    static const char*          _cpf_file_name;
};

};

#endif  //__V4L2DEV_H__
