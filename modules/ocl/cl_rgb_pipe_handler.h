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

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>

namespace XCam {

typedef struct {
    float thr_r;
    float thr_g;
    float thr_b;
    float gain;
} CLRgbPipeTnrConfig;

class CLRgbPipeImageKernel
    : public CLImageKernel
{
public:
    explicit CLRgbPipeImageKernel (const SmartPtr<CLContext> &context);
};

class CLRgbPipeImageHandler
    : public CLImageHandler
{
    typedef std::list<SmartPtr<CLImage>> CLImagePtrList;
public:
    explicit CLRgbPipeImageHandler (const SmartPtr<CLContext> &context, const char *name);
    bool set_rgb_pipe_kernel (SmartPtr<CLRgbPipeImageKernel> &kernel);
    bool set_tnr_config (const XCam3aResultTemporalNoiseReduction& config);

protected:
    virtual XCamReturn prepare_parameters (
        SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    SmartPtr<CLRgbPipeImageKernel>  _rgb_pipe_kernel;
    CLRgbPipeTnrConfig              _tnr_config;
    CLImagePtrList                  _image_in_list;
};

SmartPtr<CLImageHandler>
create_cl_rgb_pipe_image_handler (const SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_RGB_PIPE_HANLDER_H
