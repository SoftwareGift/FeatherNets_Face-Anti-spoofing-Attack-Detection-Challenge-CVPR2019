/*
 * cl_retinex_handler.h - CL retinex handler.
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
 * Author: wangfei <feix.w.wang@intel.com>
 */

#ifndef XCAM_CL_RETINEX_HANLDER_H
#define XCAM_CL_RETINEX_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"
#include "x3a_stats_pool.h"


#define XCAM_RETINEX_TABLE_SIZE 5
#define XCAM_RETINEX_SCALE 3
namespace XCam {

typedef struct {
    float           gain;
    float           threshold;
    float           log_min;
    float           log_max;
} CLRetinexConfig;

class CLRetinexImageKernel
    : public CLImageKernel
{
public:
    explicit CLRetinexImageKernel (SmartPtr<CLContext> &context);
    bool set_gaussian(int size, float sigma);
    bool get_retinex_log_value (XCam3AStats * stats);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLRetinexImageKernel);
    uint32_t _vertical_offset_in;
    uint32_t _vertical_offset_out;
    CLRetinexConfig _retinex_config;
    SmartPtr<CLBuffer>  _g_table_buffer;
    float _g_table[XCAM_RETINEX_TABLE_SIZE*XCAM_RETINEX_TABLE_SIZE];
};

class CLRetinexImageHandler
    : public CLImageHandler
{
public:
    explicit CLRetinexImageHandler (const char *name);
    bool set_retinex_kernel(SmartPtr<CLRetinexImageKernel> &kernel);
    bool set_gaussian_table(int size, float sigma);

private:
    XCAM_DEAD_COPY (CLRetinexImageHandler);
    SmartPtr<CLRetinexImageKernel> _retinex_kernel;
};

SmartPtr<CLImageHandler>
create_cl_retinex_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_RETINEX_HANLDER_H
