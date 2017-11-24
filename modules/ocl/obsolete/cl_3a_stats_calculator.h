/*
 * cl_3a_stats_calculator.h - CL 3a calculator
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CL_3A_CALCULATOR_H
#define XCAM_CL_3A_CALCULATOR_H

#include <xcam_std.h>
#include <x3a_stats_pool.h>
#include <stats_callback_interface.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_memory.h>
#include <ocl/cl_3a_stats_context.h>

namespace XCam {

class CL3AStatsCalculator;

class CL3AStatsCalculatorKernel
    : public CLImageKernel
{
public:
    explicit CL3AStatsCalculatorKernel (
        SmartPtr<CLContext> &context, SmartPtr<CL3AStatsCalculator> &image);

public:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

    virtual void pre_stop ();

private:
    bool allocate_data (const VideoBufferInfo &buffer_info);

    XCAM_DEAD_COPY (CL3AStatsCalculatorKernel);

private:
    SmartPtr<X3aStatsPool>           _stats_pool;
    SmartPtr<CLBuffer>               _stats_cl_buffer[XCAM_CL_3A_STATS_BUFFER_COUNT];
    uint32_t                         _stats_buf_index;
    SmartPtr<DrmBoBuffer>            _output_buffer;
    XCam3AStatsInfo                  _stats_info;
    bool                             _data_allocated;

    SmartPtr<CL3AStatsCalculator>    _image;
};

class CL3AStatsCalculator
    : public CLImageHandler
{
    friend class CL3AStatsCalculatorKernel;
public:
    explicit CL3AStatsCalculator ();
    void set_stats_callback (SmartPtr<StatsCallback> &callback) {
        _stats_callback = callback;
    }

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCamReturn post_stats (const SmartPtr<X3aStats> &stats);
    XCAM_DEAD_COPY (CL3AStatsCalculator);

private:
    SmartPtr<StatsCallback>         _stats_callback;
};

SmartPtr<CLImageHandler>
create_cl_3a_stats_image_handler (SmartPtr<CLContext> &context);

};

#endif // XCAM_CL_3A_CALCULATOR_H
