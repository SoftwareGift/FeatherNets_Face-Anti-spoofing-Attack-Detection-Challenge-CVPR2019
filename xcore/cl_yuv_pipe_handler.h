/*
 * cl_yuv_pipe_handler.h - CL Yuv Pipe handler
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
 * Author: Wangfei <feix.w.wang@intel.com>
 */

#ifndef XCAM_CL_YUV_PIPE_HANLDER_H
#define XCAM_CL_YUV_PIPE_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

class CLYuvPipeImageKernel
    : public CLImageKernel
{
    typedef std::list<SmartPtr<CLVaBuffer>> CLVaBufferPtrList;

public:
    explicit CLYuvPipeImageKernel (SmartPtr<CLContext> &context);
    bool set_macc (const XCam3aResultMaccMatrix &macc);
    bool set_matrix (const XCam3aResultColorMatrix &matrix);
    bool set_tnr_yuv_config (const XCam3aResultTemporalNoiseReduction& config);
    bool set_tnr_enable (bool enable_tnr_yuv);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);
    virtual XCamReturn post_execute ();

private:
    XCAM_DEAD_COPY (CLYuvPipeImageKernel);
    SmartPtr<CLBuffer>  _matrix_buffer;
    SmartPtr<CLBuffer>  _macc_table_buffer;
    float               _macc_table[XCAM_CHROMA_AXIS_SIZE * XCAM_CHROMA_MATRIX_SIZE];
    float               _rgbtoyuv_matrix[XCAM_COLOR_MATRIX_SIZE];
    uint32_t            _vertical_offset;
    uint32_t            _plannar_offset;
    float               _gain_yuv;
    float               _thr_y;
    float               _thr_uv;
    uint32_t            _enable_tnr_yuv;
    uint32_t            _enable_tnr_yuv_state;
    SmartPtr<CLVaBuffer> _image_in;
    SmartPtr<CLVaBuffer> _image_out;
    SmartPtr<CLVaBuffer> _image_out_prev;
};

class CLYuvPipeImageHandler
    : public CLImageHandler
{
public:
    explicit CLYuvPipeImageHandler (const char *name);
    bool set_yuv_pipe_kernel(SmartPtr<CLYuvPipeImageKernel> &kernel);
    bool set_macc_table (const XCam3aResultMaccMatrix &macc);
    bool set_rgbtoyuv_matrix (const XCam3aResultColorMatrix &matrix);
    bool set_tnr_yuv_config (const XCam3aResultTemporalNoiseReduction& config);
    bool set_tnr_enable (bool enable_tnr_yuv);

protected:
    virtual XCamReturn prepare_buffer_pool_video_info (
        const VideoBufferInfo &input,
        VideoBufferInfo &output);

private:
    XCAM_DEAD_COPY (CLYuvPipeImageHandler);
    SmartPtr<CLYuvPipeImageKernel> _yuv_pipe_kernel;
    uint32_t  _output_format;
};

SmartPtr<CLImageHandler>
create_cl_yuv_pipe_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_YUV_PIPE_HANLDER_H
