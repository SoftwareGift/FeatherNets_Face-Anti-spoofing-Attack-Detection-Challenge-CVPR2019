/*
  * main_pipe_manager.h -main pipe manager
  *
  *  Copyright (c) 2016 Intel Corporation
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
  * Author: Yinhang Liu <yinhangx.liu@intel.com>
  */

#ifndef XCAMFILTER_MAIN_PIPE_MANAGER_H
#define XCAMFILTER_MAIN_PIPE_MANAGER_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <pipe_manager.h>
#include <video_buffer.h>
#include <smart_analyzer_loader.h>
#include <ocl/cl_post_image_processor.h>

namespace GstXCam {

class MainPipeManager
    : public XCam::PipeManager
{
public:
    MainPipeManager () {};
    ~MainPipeManager () {};

    XCam::SmartPtr<XCam::VideoBuffer> dequeue_buffer (const int32_t timeout);
    void pause_dequeue ();
    void resume_dequeue ();

    void set_image_processor (XCam::SmartPtr<XCam::CLPostImageProcessor> &processor) {
        _image_processor = processor;
    }

    XCam::SmartPtr<XCam::CLPostImageProcessor> &get_image_processor () {
        return _image_processor;
    }

protected:
    virtual void post_buffer (const XCam::SmartPtr<XCam::VideoBuffer> &buf);

private:
    XCam::SafeList<XCam::VideoBuffer>           _ready_buffers;
    XCam::SmartPtr<XCam::CLPostImageProcessor>  _image_processor;
};

};

#endif // XCAMFILTER_MAIN_PIPE_MANAGER_H
