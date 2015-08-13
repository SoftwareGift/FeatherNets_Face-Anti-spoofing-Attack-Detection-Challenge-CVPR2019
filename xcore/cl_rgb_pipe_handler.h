/*
 * cl_rgb_pipe_handler.h - CL rgb pipe handler
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
 * Author: Shincy Tu <shincy.tu@intel.com>
 * Author: Wei Zong <wei.zong@intel.com>
 * Author: Wangfei <feix.w.wang@intel.com>
 */

#ifndef XCAM_CL_RGB_PIPE_HANLDER_H
#define XCAM_CL_RGB_PIPE_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"

namespace XCam {

enum CLTgbPipeTnrLightCondition {
    CL_RGBPIPE_TNR_LOW_LIGHT = 0,
    CL_RGBPIPE_TNR_INDOOR    = 1,
    CL_RGBPIPE_TNR_DAY_LIGHT = 2,
    CL_RGBPIPE_TNR_LIGHT_COUNT
};

typedef struct {
    float thr_r;
    float thr_g;
    float thr_b;
} CLRgbPipeTnrConfig;

static const CLRgbPipeTnrConfig rgbpipe_tnr_threshold[CL_RGBPIPE_TNR_LIGHT_COUNT] = {
    {0.0642, 0.0451, 0.0733}, // low light R/G/B/threshold
    {0.0045, 0.0029, 0.0039}, // Indoor R/G/B/ threshold
    {0.0032, 0.0029, 0.0030}  // Daylight R/G/B/ threshold
};

class CLRgbPipeImageKernel
    : public CLImageKernel
{
    typedef std::list<SmartPtr<CLImage>> CLImagePtrList;
public:
    explicit CLRgbPipeImageKernel (SmartPtr<CLContext> &context);
    virtual ~CLRgbPipeImageKernel () {
        _image_in_list.clear ();
    }
    bool set_tnr_threshold (float r, float g, float b);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLRgbPipeImageKernel);
    CLRgbPipeTnrConfig _tnr_config;
    CLImagePtrList _image_in_list;
};

class CLRgbPipeImageHandler
    : public CLImageHandler
{
public:
    explicit CLRgbPipeImageHandler (const char *name);
    bool set_rgb_pipe_kernel (SmartPtr<CLRgbPipeImageKernel> &kernel);
    bool set_tnr_exposure_params (double a_gain, double d_gain, int32_t exposure_time);

private:
    XCAM_DEAD_COPY (CLRgbPipeImageHandler);
    SmartPtr<CLRgbPipeImageKernel> _rgb_pipe_kernel;
};

SmartPtr<CLImageHandler>
create_cl_rgb_pipe_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_RGB_PIPE_HANLDER_H
