/*
 * cl_image_warp_handler.h - CL image warping handler
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_CL_IMAGE_WARP_H
#define XCAM_CL_IMAGE_WARP_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "cl_memory.h"

namespace XCam {

typedef struct {
    int frame_id;
    int valid;
    int width;
    int height;
    float trim_ratio;
    float proj_mat[9];
} CLWarpConfig;

class CLImageWarpHandler;

class CLImageWarpKernel
    : public CLImageKernel
{
    typedef std::list<SmartPtr<CLImage>> CLImagePtrList;

public:
    explicit CLImageWarpKernel (SmartPtr<CLContext> &context,
                                const char *name,
                                uint32_t channel,
                                SmartPtr<CLImageWarpHandler> &handler);

    virtual ~CLImageWarpKernel () {
        _image_in_list.clear ();
    }

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);

public:

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLImageWarpKernel);

    uint32_t _channel;
    SmartPtr<CLImageWarpHandler> _handler;
    int32_t _input_frame_id;
    int32_t _warp_frame_id;
    CLWarpConfig _warp_config;
    CLImagePtrList _image_in_list;
};

class CLImageTrimKernel
    : public CLImageKernel
{
public:
    explicit CLImageTrimKernel (SmartPtr<CLContext> &context,
                                const char *name,
                                uint32_t channel,
                                float trim_ratio,
                                SmartPtr<CLImageWarpHandler> &handler);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLImageTrimKernel);

    uint32_t _channel;
    float _trim_ratio;
    SmartPtr<CLImageWarpHandler> _handler;
};

class CLImageWarpHandler
    : public CLImageHandler
{
public:
    explicit CLImageWarpHandler ();

    bool set_warp_config (const XCamDVSResult* config);
    const CLWarpConfig& get_warp_config () const {
        return _warp_config;
    };

private:
    XCAM_DEAD_COPY (CLImageWarpHandler);

    void reset_projection_matrix ();

    CLWarpConfig _warp_config;
};

SmartPtr<CLImageHandler>
create_cl_image_warp_handler (SmartPtr<CLContext> &context);

};

#endif // XCAM_CL_IMAGE_WARP_H
