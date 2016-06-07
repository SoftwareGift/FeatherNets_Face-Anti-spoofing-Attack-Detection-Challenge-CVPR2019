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

enum CLWaveletSubband {
    CL_WAVELET_SUBBAND_LL = 0,
    CL_WAVELET_SUBBAND_HL,
    CL_WAVELET_SUBBAND_LH,
    CL_WAVELET_SUBBAND_HH,
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
    uint32_t channel;
    int32_t layer;
    float noise_variance[3];
    SmartPtr<CLImage> ll;
    SmartPtr<CLImage> hl[3];
    SmartPtr<CLImage> lh[3];
    SmartPtr<CLImage> hh[3];
} CLWaveletDecompBuffer;

class CLNewWaveletDenoiseImageHandler;

class CLWaveletNoiseEstimateKernel
    : public CLImageKernel
{

public:
    explicit CLWaveletNoiseEstimateKernel (SmartPtr<CLContext> &context,
                                           const char *name,
                                           SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
                                           uint32_t channel, uint32_t subband, uint32_t layer);

    SmartPtr<CLImage> get_input_buffer (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);
    SmartPtr<CLImage> get_output_buffer (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

    XCamReturn estimate_noise_variance (const VideoBufferInfo & video_info, SmartPtr<CLImage> image, float* noise_var);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLWaveletNoiseEstimateKernel);

private:
    uint32_t  _decomposition_levels;
    uint32_t  _channel;
    uint32_t  _subband;
    uint32_t  _current_layer;
    float     _analog_gain;

    SmartPtr<CLNewWaveletDenoiseImageHandler> _handler;
};

class CLWaveletThresholdingKernel
    : public CLImageKernel
{

public:
    explicit CLWaveletThresholdingKernel (SmartPtr<CLContext> &context,
                                          const char *name,
                                          SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
                                          uint32_t channel, uint32_t layer);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLWaveletThresholdingKernel);

private:
    uint32_t  _decomposition_levels;
    uint32_t  _channel;
    uint32_t  _current_layer;
    float     _hard_threshold;
    float     _soft_threshold;
    float     _anolog_gain_weight;
    SmartPtr<CLNewWaveletDenoiseImageHandler> _handler;
    float     _noise_variance[2];
};

class CLWaveletTransformKernel
    : public CLImageKernel
{

public:
    explicit CLWaveletTransformKernel (SmartPtr<CLContext> &context,
                                       const char *name,
                                       SmartPtr<CLNewWaveletDenoiseImageHandler> &handler,
                                       CLWaveletFilterBank fb,
                                       uint32_t channel,
                                       uint32_t layer);

    SmartPtr<CLWaveletDecompBuffer> get_decomp_buffer (uint32_t channel, int layer);

protected:
    virtual XCamReturn prepare_arguments (
        SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output,
        CLArgument args[], uint32_t &arg_count,
        CLWorkSize &work_size);

private:
    XCAM_DEAD_COPY (CLWaveletTransformKernel);

    SmartPtr<CLImage> _image_in_uv;
    SmartPtr<CLImage> _image_out_uv;

    CLWaveletFilterBank _filter_bank;
    uint32_t  _decomposition_levels;
    uint32_t  _channel;
    uint32_t  _current_layer;
    float     _hard_threshold;
    float     _soft_threshold;

    SmartPtr<CLNewWaveletDenoiseImageHandler> _handler;
};

class CLNewWaveletDenoiseImageHandler
    : public CLCloneImageHandler
{
    typedef std::list<SmartPtr<CLWaveletDecompBuffer>> CLWaveletDecompBufferList;

public:
    explicit CLNewWaveletDenoiseImageHandler (const char *name, uint32_t channel);

    bool set_denoise_config (const XCam3aResultWaveletNoiseReduction& config);
    XCam3aResultWaveletNoiseReduction& get_denoise_config () {
        return _config;
    };

    SmartPtr<CLWaveletDecompBuffer> get_decomp_buffer (uint32_t channel, int layer);

    void set_estimated_noise_variation (float* noise_var);
    void get_estimated_noise_variation (float* noise_var);

    void dump_coeff (SmartPtr<CLImage> image, uint32_t channel, uint32_t layer, uint32_t subband);

protected:
    virtual XCamReturn prepare_output_buf (SmartPtr<DrmBoBuffer> &input, SmartPtr<DrmBoBuffer> &output);

private:
    XCAM_DEAD_COPY (CLNewWaveletDenoiseImageHandler);

private:
    uint32_t _channel;
    XCam3aResultWaveletNoiseReduction _config;
    CLWaveletDecompBufferList _decompBufferList;
    float _noise_variance[3];
};

SmartPtr<CLImageHandler>
create_cl_newwavelet_denoise_image_handler (SmartPtr<CLContext> &context, uint32_t channel);

};

#endif //XCAM_CL_NEWWAVELET_DENOISE_HANLDER_H
