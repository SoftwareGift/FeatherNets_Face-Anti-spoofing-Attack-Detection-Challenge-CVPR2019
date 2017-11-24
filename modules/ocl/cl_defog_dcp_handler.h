/*
 * cl_defog_dcp_handler.h - CL defog dark channel prior handler
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CL_DEFOG_DCP_HANLDER_H
#define XCAM_CL_DEFOG_DCP_HANLDER_H

#include <xcam_std.h>
#include <base/xcam_3a_result.h>
#include <x3a_stats_pool.h>
#include <ocl/cl_image_handler.h>

#define XCAM_DEFOG_DC_ORIGINAL      0
#define XCAM_DEFOG_DC_MIN_FILTER_V  1
#define XCAM_DEFOG_DC_MIN_FILTER_H  2
#define XCAM_DEFOG_DC_BI_FILTER     3
#define XCAM_DEFOG_DC_REFINED       4
#define XCAM_DEFOG_DC_MAX_BUF       5


#define XCAM_DEFOG_R_CHANNEL    0
#define XCAM_DEFOG_G_CHANNEL    1
#define XCAM_DEFOG_B_CHANNEL    2
#define XCAM_DEFOG_MAX_CHANNELS 3

namespace XCam {

class CLDefogDcpImageHandler;

class CLDarkChannelKernel
    : public CLImageKernel
{
public:
    explicit CLDarkChannelKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> &defog_handler);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    SmartPtr<CLDefogDcpImageHandler>   _defog_handler;
};

class CLMinFilterKernel
    : public CLImageKernel
{
public:
    explicit CLMinFilterKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> &defog_handler, int index);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

    SmartPtr<CLDefogDcpImageHandler>   _defog_handler;
    uint32_t                           _buf_index;
};

class CLBiFilterKernel
    : public CLImageKernel
{
public:
    explicit CLBiFilterKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> &defog_handler);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLBiFilterKernel);

private:
    SmartPtr<CLDefogDcpImageHandler>   _defog_handler;
};

class CLDefogRecoverKernel
    : public CLImageKernel
{
public:
    explicit CLDefogRecoverKernel (
        const SmartPtr<CLContext> &context, SmartPtr<CLDefogDcpImageHandler> &defog_handler);

protected:
    virtual XCamReturn prepare_arguments (CLArgList &args, CLWorkSize &work_size);

private:
    float get_max_value (SmartPtr<VideoBuffer> &buf);

    XCAM_DEAD_COPY (CLDefogRecoverKernel);

private:
    SmartPtr<CLDefogDcpImageHandler>   _defog_handler;
    float                              _max_r;
    float                              _max_g;
    float                              _max_b;
    float                              _max_i;
};

class CLDefogDcpImageHandler
    : public CLImageHandler
{
public:
    explicit CLDefogDcpImageHandler (
        const SmartPtr<CLContext> &context, const char *name);

    SmartPtr<CLImage> &get_dark_map (uint index) {
        XCAM_ASSERT (index < XCAM_DEFOG_DC_MAX_BUF);
        return _dark_channel_buf[index];
    };
    SmartPtr<CLImage> &get_rgb_channel (uint index) {
        XCAM_ASSERT (index < XCAM_DEFOG_MAX_CHANNELS);
        return _rgb_buf[index];
    };

protected:
    virtual XCamReturn prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);
    virtual XCamReturn execute_done (SmartPtr<VideoBuffer> &output);

private:
    XCamReturn allocate_transmit_bufs (const VideoBufferInfo &video_info);
    void dump_buffer();

    XCAM_DEAD_COPY (CLDefogDcpImageHandler);

private:
    SmartPtr<CLImage>                 _dark_channel_buf[XCAM_DEFOG_DC_MAX_BUF];
    SmartPtr<CLImage>                 _rgb_buf[XCAM_DEFOG_MAX_CHANNELS];
};

SmartPtr<CLImageHandler>
create_cl_defog_dcp_image_handler (const SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_DEFOG_DCP_HANLDER_H
