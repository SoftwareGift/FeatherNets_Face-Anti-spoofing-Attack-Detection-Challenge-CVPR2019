/*
 * cl_image_warp_handler.h - CL image warping handler
 *
 *  Copyright (c) 2016 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_CL_IMAGE_WARP_H
#define XCAM_CL_IMAGE_WARP_H

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>
#include <ocl/cl_memory.h>

namespace XCam {

#define CL_IMAGE_WARP_WRITE_UINT 1

enum {
#if CL_IMAGE_WARP_WRITE_UINT
    KernelImageWarp   = 0,
#else
    KernelImageWarp   = 1,
#endif
};

struct CLWarpConfig {
    int frame_id;
    int width;
    int height;
    float trim_ratio;
    float proj_mat[9];

    CLWarpConfig ()
        : frame_id (-1)
        , width (-1)
        , height (-1)
        , trim_ratio (0.05f)
    {
        proj_mat[0] = 1.0f;
        proj_mat[1] = 0.0f;
        proj_mat[2] = 0.0f;
        proj_mat[3] = 0.0f;
        proj_mat[4] = 1.0f;
        proj_mat[5] = 0.0f;
        proj_mat[6] = 0.0f;
        proj_mat[7] = 0.0f;
        proj_mat[8] = 1.0f;
    };
};

class CLImageWarpHandler;

class CLImageWarpKernel
    : public CLImageKernel
{
public:
    explicit CLImageWarpKernel (
        const SmartPtr<CLContext> &context,
        const char *name,
        uint32_t channel,
        SmartPtr<CLImageHandler> &handler);

    virtual ~CLImageWarpKernel () {};

protected:
    virtual XCamReturn prepare_arguments (
        CLArgList &args, CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLImageWarpKernel);

    uint32_t                     _channel;
    SmartPtr<CLImageWarpHandler> _handler;
};

class CLImageWarpHandler
    : public CLImageHandler
{
    typedef std::list<CLWarpConfig> CLWarpConfigList;

public:
    explicit CLImageWarpHandler (const SmartPtr<CLContext> &context, const char *name = "CLImageWarpHandler");
    virtual ~CLImageWarpHandler () {
        _warp_config_list.clear ();
    }

    virtual SmartPtr<VideoBuffer> get_warp_input_buf ();

    bool set_warp_config (const XCamDVSResult& config);
    CLWarpConfig get_warp_config ();

    virtual bool is_ready ();

protected:
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLImageWarpHandler);

    CLWarpConfigList _warp_config_list;

};

SmartPtr<CLImageHandler>
create_cl_image_warp_handler (const SmartPtr<CLContext> &context);

};

#endif // XCAM_CL_IMAGE_WARP_H
