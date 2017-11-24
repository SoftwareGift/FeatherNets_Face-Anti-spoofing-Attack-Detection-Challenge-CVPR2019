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

#include <xcam_std.h>
#include <ocl/cl_image_handler.h>
#include <base/xcam_3a_result.h>

namespace XCam {

class CLWaveletDenoiseImageHandler;

class CLWaveletDenoiseImageKernel
    : public CLImageKernel
{

private:

public:
    explicit CLWaveletDenoiseImageKernel (
        const SmartPtr<CLContext> &context,
        const char *name,
        SmartPtr<CLWaveletDenoiseImageHandler> &handler,
        uint32_t channel,
        uint32_t layer);

    virtual ~CLWaveletDenoiseImageKernel () {
    }

protected:
    virtual XCamReturn prepare_arguments (
        CLArgList &args, CLWorkSize &work_size);

private:
    uint32_t  _channel;
    uint32_t  _current_layer;

    SmartPtr<CLWaveletDenoiseImageHandler> _handler;
};

class CLWaveletDenoiseImageHandler
    : public CLImageHandler
{
public:
    explicit CLWaveletDenoiseImageHandler (const SmartPtr<CLContext> &context, const char *name);

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
    virtual XCamReturn prepare_output_buf (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output);

private:
    XCam3aResultWaveletNoiseReduction _config;
    SmartPtr<CLMemory> _details_image;
    SmartPtr<CLMemory> _approx_image;
};

SmartPtr<CLImageHandler>
create_cl_wavelet_denoise_image_handler (const SmartPtr<CLContext> &context, uint32_t channel);

};

#endif //XCAM_CL_WAVELET_DENOISE_HANLDER_H
