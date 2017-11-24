/*
 * soft_copy_task.h - soft copy class
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_SOFT_COPY_TASK_H
#define XCAM_SOFT_COPY_TASK_H

#include <xcam_std.h>
#include <soft/soft_worker.h>
#include <soft/soft_handler.h>
#include <soft/soft_image.h>
#include <interface/stitcher.h>

namespace XCam {

namespace XCamSoftTasks {

class CopyTask
    : public SoftWorker
{
public:
    struct Args : SoftArgs {
        SmartPtr<UcharImage>         in_luma, out_luma;
        SmartPtr<Uchar2Image>        in_uv, out_uv;

        Args (const SmartPtr<ImageHandler::Parameters> &param)
            : SoftArgs (param)
        {}
    };

public:
    explicit CopyTask (const SmartPtr<Worker::Callback> &cb)
        : SoftWorker ("CopyTask", cb)
    {}

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
};

}

}

#endif // XCAM_SOFT_COPY_TASK_H