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

#ifndef XCAM_SOFT_GEO_TASKS_PRIV_H
#define XCAM_SOFT_GEO_TASKS_PRIV_H

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

class GeoMapDualConstTask
    : public GeoMapTask
{
public:
    struct Args : GeoMapTask::Args {
        Float2    left_factor;
        Float2    right_factor;

        Args (
            const SmartPtr<ImageHandler::Parameters> &param)
            : GeoMapTask::Args (param)
        {}
    };

public:
    explicit GeoMapDualConstTask (const SmartPtr<Worker::Callback> &cb)
        : GeoMapTask (cb)
    {
        set_work_uint (8, 2);
    }

private:
    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);
};

class GeoMapDualCurveTask
    : public GeoMapDualConstTask
{
public:
    struct Args : GeoMapDualConstTask::Args {
        Args (
            const SmartPtr<ImageHandler::Parameters> &param)
            : GeoMapDualConstTask::Args (param)
        {}
    };

public:
    explicit GeoMapDualCurveTask (const SmartPtr<Worker::Callback> &cb);
    ~GeoMapDualCurveTask ();

    void set_scaled_height (float scaled_height) {
        XCAM_ASSERT (!XCAM_DOUBLE_EQUAL_AROUND (scaled_height, 0.0f));
        _scaled_height = scaled_height;
    }

    void set_left_std_factor (float x, float y);
    void set_right_std_factor (float x, float y);

private:
    void set_factors (SmartPtr<GeoMapDualCurveTask::Args> args, uint32_t size);
    bool set_steps (uint32_t size);

    virtual XCamReturn work_range (const SmartPtr<Arguments> &args, const WorkRange &range);

    XCAM_DEAD_COPY (GeoMapDualCurveTask);

private:
    float        _scaled_height;
    Float2       _left_std_factor;
    Float2       _right_std_factor;
    Float2       *_left_factors;
    Float2       *_right_factors;
    Float2       *_left_steps;
    Float2       *_right_steps;
};

}

}
#endif // XCAM_SOFT_GEO_TASKS_PRIV_H
