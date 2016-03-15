/*
 * main_dev_manager.h - main device manager
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

#ifndef XCAMSRC_MAIN_DEV_MANAGER_H
#define XCAMSRC_MAIN_DEV_MANAGER_H

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif
#include <base/xcam_common.h>
#include <linux/videodev2.h>
#include <linux/atomisp.h>
#include <stdint.h>
#include <unistd.h>

#include <gst/video/video.h>
#include <gst/gst.h>

#include <queue>

#include <xcam_mutex.h>
#include <video_buffer.h>
#include <v4l2_buffer_proxy.h>
#include <v4l2_device.h>
#include <device_manager.h>
#include <atomisp_device.h>
#include <device_manager.h>
#include <isp_controller.h>
#include <isp_image_processor.h>
#if HAVE_LIBCL
#include <cl_3a_image_processor.h>
#include <cl_post_image_processor.h>
#endif
#if HAVE_IA_AIQ
#include <x3a_analyzer_aiq.h>
#endif
#include <x3a_analyzer_simple.h>

namespace GstXCam {

class MainDeviceManager;

class MainDeviceManager
    : public XCam::DeviceManager
{
public:
    MainDeviceManager ();
    ~MainDeviceManager ();

    XCam::SmartPtr<XCam::VideoBuffer> dequeue_buffer ();
    void pause_dequeue ();
    void resume_dequeue ();

#if HAVE_LIBCL
public:
    void set_cl_image_processor (XCam::SmartPtr<XCam::CL3aImageProcessor> &processor) {
        _cl_image_processor = processor;
    }

    XCam::SmartPtr<XCam::CL3aImageProcessor> &get_cl_image_processor () {
        return _cl_image_processor;
    }

    void set_cl_post_image_processor (XCam::SmartPtr<XCam::CLPostImageProcessor> &processor) {
        _cl_post_image_processor = processor;
    }

    XCam::SmartPtr<XCam::CLPostImageProcessor> &get_cl_post_image_processor () {
        return _cl_post_image_processor;
    }
#endif

protected:
    virtual void handle_message (const XCam::SmartPtr<XCam::XCamMessage> &msg);
    virtual void handle_buffer (const XCam::SmartPtr<XCam::VideoBuffer> &buf);

private:
    XCam::SafeList<XCam::VideoBuffer>         _ready_buffers;
#if HAVE_LIBCL
    XCam::SmartPtr<XCam::CL3aImageProcessor>   _cl_image_processor;
    XCam::SmartPtr<XCam::CLPostImageProcessor> _cl_post_image_processor;
#endif
};

};

#endif  //XCAMSRC_MAIN_DEV_MANAGER_H
