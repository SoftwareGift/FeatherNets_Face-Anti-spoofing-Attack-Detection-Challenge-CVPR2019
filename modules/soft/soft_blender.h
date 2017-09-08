/*
 * soft_blender.h - soft blender class
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_SOFT_BLENDER_H
#define XCAM_SOFT_BLENDER_H

#include "xcam_utils.h"
#include "interface/blender.h"
#include "soft/soft_handler.h"

namespace XCam {

class SoftBlender
    : public SoftHandler, public Blender
{
    friend SmartPtr<SoftHandler> create_soft_blender ();

public:
    ~SoftBlender ();

    XCamReturn gauss_scale_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

protected:
    explicit SoftBlender (const char *name = "SoftBlender");

    //derived from Blender interface
    XCamReturn blend (
        const SmartPtr<VideoBuffer> &in0,
        const SmartPtr<VideoBuffer> &in1,
        SmartPtr<VideoBuffer> &out_buf);

    //derived from SoftHandler
    virtual SmartPtr<Worker::Arguments> get_first_worker_args (
        const SmartPtr<SoftWorker> &worker, SmartPtr<ImageHandler::Parameters> &params);

private:
};

extern SmartPtr<SoftHandler> create_soft_blender ();
}

#endif //XCAM_SOFT_BLENDER_H
