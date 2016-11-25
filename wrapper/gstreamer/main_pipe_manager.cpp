/*
  * main_pipe_manager.cpp -main pipe manager
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

#include "main_pipe_manager.h"

using namespace XCam;

namespace GstXCam {

void
MainPipeManager::post_buffer (const SmartPtr<VideoBuffer> &buf)
{
    XCAM_ASSERT (buf.ptr ());
    _ready_buffers.push (buf);
}

SmartPtr<VideoBuffer>
MainPipeManager::dequeue_buffer (const int32_t timeout)
{
    SmartPtr<VideoBuffer> ret;
    ret = _ready_buffers.pop (timeout);
    return ret;
}

void
MainPipeManager::pause_dequeue ()
{
    return _ready_buffers.pause_pop ();
}

void
MainPipeManager::resume_dequeue ()
{
    return _ready_buffers.resume_pop ();
}

};
