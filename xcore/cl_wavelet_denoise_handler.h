/*
 * cl_wavelet_denoise_handler.h - CL wavelet denoise handler
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

#ifndef XCAM_CL_WAVELET_DENOISE_HANLDER_H
#define XCAM_CL_WAVELET_DENOISE_HANLDER_H

#include "xcam_utils.h"
#include "cl_image_handler.h"
#include "base/xcam_3a_result.h"

namespace XCam {

#define WAVELET_DECOMPOSITION_LEVELS 4

class CLWaveletDenoiseImageHandler;

class CLWaveletDenoiseImageKernel
    : public CLImageKernel
{

private:

public:
    explicit CLWaveletDenoiseImageKernel (SmartPtr<CLContext> &context,
                                          const char *name,
                                          SmartPtr<CLWaveletDenoiseImageHandler> &handler,
                                          uint32_t layer);

    virtual ~CLWaveletDenoiseImageKernel () {
    }

    virtual XCamReturn post_execute (SmartPtr<DrmBoBuffer> &output);
protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLWaveletDenoiseImageKernel);

    float     _hard_threshold;
    float     _soft_threshold;
    uint32_t  _decomposition_levels;
    uint32_t  _current_layer;
    uint32_t  _input_y_offset;
    uint32_t  _output_y_offset;
    uint32_t  _input_uv_offset;
    uint32_t  _output_uv_offset;

    SmartPtr<CLWaveletDenoiseImageHandler> _handler;

    SmartPtr<CLMemory>  _input_image;
    SmartPtr<CLMemory>  _approx_image;
    SmartPtr<CLMemory>  _details_image;
    SmartPtr<CLMemory>  _reconstruct_image;
};

class CLWaveletDenoiseImageHandler
    : public CLImageHandler
{
public:
    explicit CLWaveletDenoiseImageHandler (const char *name);

    bool set_denoise_config (const XCam3aResultWaveletNoiseReduction& config);
    XCam3aResultWaveletNoiseReduction& get_denoise_config () {
        return _config;
    };

    SmartPtr<CLMemory> &get_details_image () {
        return _details_image;
    };

    SmartPtr<CLMemory> &get_approx_image () {
        return _approx_image;
    };

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLWaveletDenoiseImageHandler);

private:
    XCam3aResultWaveletNoiseReduction _config;
    SmartPtr<CLMemory> _details_image;
    SmartPtr<CLMemory> _approx_image;
};

SmartPtr<CLImageHandler>
create_cl_wavelet_denoise_image_handler (SmartPtr<CLContext> &context);

};

#endif //XCAM_CL_WAVELET_DENOISE_HANLDER_H
