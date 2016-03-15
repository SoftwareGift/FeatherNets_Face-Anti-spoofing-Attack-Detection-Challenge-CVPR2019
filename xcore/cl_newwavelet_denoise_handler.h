/*
 * cl_newwavelet_denoise_handler.h - CL wavelet denoise handler
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
 * Author: Wei Zong <wei.zong@intel.com>
 */

#ifndef XCAM_CL_NEWWAVELET_DENOISE_HANLDER_H
#define XCAM_CL_NEWWAVELET_DENOISE_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

enum CLWaveletFilterBank {
    CL_WAVELET_HAAR_ANALYSIS = 0,
    CL_WAVELET_HAAR_SYNTHESIS = 1,
};

/*------------------------
 Wavelet decomposition
     frequency block

              __ width__
     ___________________
    |         |         |  |
    |         |         |  |
    |  LL     |  HL     |height
    |         |         |  |
    |_________|_________|  |
    |         |         |
    |         |         |
    |  LH     |  HH     |
    |         |         |
    |_________|_________|
--------------------------*/
typedef struct _CLCLWaveletDecompBuffer {
    int32_t width;
    int32_t height;
    int32_t layer;
    SmartPtr<CLImage> ll;
    SmartPtr<CLImage> hl;
    SmartPtr<CLImage> lh;
    SmartPtr<CLImage> hh;
} CLWaveletDecompBuffer;

class CLNewWaveletDenoiseImageHandler;

class CLNewWaveletDenoiseImageKernel
    : public CLImageKernel
{

private:

public:
    explicit CLNewWaveletDenoiseImageKernel (SmartPtr<CLContext> &context,
            const char *name,
            SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
            CLWaveletFilterBank fb,
            uint32_t layer);

    virtual ~CLNewWaveletDenoiseImageKernel () {
    }

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
    SmartPtr<CLWaveletDecompBuffer> get_decomp_buffer (int layer);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLNewWaveletDenoiseImageKernel);

    CLWaveletFilterBank _filter_bank;
    uint32_t  _decomposition_levels;
    uint32_t  _current_layer;
    float     _hard_threshold;
    float     _soft_threshold;
    uint32_t  _input_y_offset;
    uint32_t  _output_y_offset;
    uint32_t  _input_uv_offset;
    uint32_t  _output_uv_offset;

    SmartPtr<CLNewWaveletDenoiseImageHandler> _handler;
};

class CLNewWaveletDenoiseImageHandler
    : public CLCloneImageHandler
{
    typedef std::list<SmartPtr<CLWaveletDecompBuffer>> CLWaveletDecompBufferList;

public:
    explicit CLNewWaveletDenoiseImageHandler (const char *name);

    bool set_denoise_config (const XCam3aResultWaveletNoiseReduction& config);
    XCam3aResultWaveletNoiseReduction& get_denoise_config () {
        return _config;
    };

    SmartPtr<CLWaveletDecompBuffer> get_decomp_buffer (int layer);

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLNewWaveletDenoiseImageHandler);

private:
    XCam3aResultWaveletNoiseReduction _config;
    CLWaveletDecompBufferList _decompBufferList;
};

SmartPtr<CLImageHandler>
create_cl_newwavelet_denoise_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_NEWWAVELET_DENOISE_HANLDER_H
