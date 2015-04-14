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

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "cl_memory.h"
#include "x3a_stats_pool.h"

namespace XCam {

class CL3AStatsCalculatorKernel
    : public CLImageKernel
{
public:
    explicit CL3AStatsCalculatorKernel (SmartPtr<CLContext> &context);

public:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

    virtual XCamReturn post_execute ();

private:
    bool allocate_data (const VideoBufferInfo &buffer_info);
    XCamReturn post_stats (const SmartPtr<X3aStats> &stats);

    XCAM_DEAD_COPY (CL3AStatsCalculatorKernel);

private:
    SmartPtr<X3aStatsPool>           _stats_pool;
    SmartPtr<CLBuffer>               _stats_cl_buffer;
    XCam3AStatsInfo                  _stats_info;
    bool                             _data_allocated;
};

class CL3AStatsCalculator
    : public CLImageHandler
{
public:
    explicit CL3AStatsCalculator ();

public:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CL3AStatsCalculator);
};

SmartPtr<CLImageHandler>
create_cl_3a_stats_image_handler (SmartPtr<CLContext> &context);

};

#endif // XCAM_CL_3A_CALCULATOR_H
