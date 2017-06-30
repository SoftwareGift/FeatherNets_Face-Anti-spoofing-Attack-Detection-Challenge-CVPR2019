/*
 * cl_demosaic_handler.h - CL demosaic handler
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

#ifndef XCAM_CL_DEMOSAIC_HANLDER_H
#define XCAM_CL_DEMOSAIC_HANLDER_H

#include "xcam_utils.h"
#include "ocl/cl_image_handler.h"

namespace XCam {

class CLDemosaicImageKernel
    : public CLImageKernel
{
public:
    explicit CLDemosaicImageKernel (SmartPtr<CLContext> &context);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLDemosaicImageKernel);
};

class CLBayer2RGBImageHandler
    : public CLImageHandler
{
public:
    explicit CLBayer2RGBImageHandler (const char *name);
    bool set_output_format (uint32_t fourcc);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCAM_DEAD_COPY (CLBayer2RGBImageHandler);

private:
    uint32_t  _output_format;
};

SmartPtr<CLImageHandler>
create_cl_demosaic_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_DEMOSAIC_HANLDER_H