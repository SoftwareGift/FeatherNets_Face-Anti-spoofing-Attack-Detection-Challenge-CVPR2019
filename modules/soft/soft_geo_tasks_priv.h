/*
 * soft_geo_tasks_priv.h - soft geometry map tasks
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

#include <xcam_std.h>
#include <soft/soft_worker.h>
#include <soft/soft_image.h>
#include <soft/soft_handler.h>

namespace XCam {

namespace XCamSoftTasks {

class GeoMapTask
    : public SoftWorker
{
public:
    struct Args : SoftArgs {
        SmartPtr<UcharImage>        in_luma, out_luma;
        SmartPtr<Uchar2Image>       in_uv, out_uv;
        SmartPtr<Float2Image>       lookup_table;
        Float2                      factors;

        Args (
            const SmartPtr<ImageHandler::Parameters> &param)
            : SoftArgs (param)
        {}
    };

public:
    explicit GeoMapTask (const SmartPtr<Worker::Callback> &cb)
        : SoftWorker ("GeoMapTask", cb)
    {
        set_work_uint (8, 2);
    }

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
};

}

}
