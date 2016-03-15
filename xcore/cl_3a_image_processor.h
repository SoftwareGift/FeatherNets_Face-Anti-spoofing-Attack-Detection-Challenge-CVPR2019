/*
 * cl_3a_image_processor.h - CL 3A image processor
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

#ifndef XCAM_CL_3A_IMAGE_PROCESSOR_H
#define XCAM_CL_3A_IMAGE_PROCESSOR_H

#include "xcam_utils.h"
#include <base/xcam_3a_types.h>
#include "cl_image_processor.h"
#include "stats_callback_interface.h"

namespace XCam {

class CLCscImageHandler;
class CLEeImageHandler;
class CLBayerBasicImageHandler;
class CLBayerPipeImageHandler;
class CLYuvPipeImageHandler;
class CLTonemappingImageHandler;
class CLNewTonemappingImageHandler;
class CLImageScaler;
class CLWaveletDenoiseImageHandler;
class CLNewWaveletDenoiseImageHandler;

#define ENABLE_YEENR_HANDLER 0

class CL3aImageProcessor
    : public CLImageProcessor
{
public:
    enum OutSampleType {
        OutSampleYuv,
        OutSampleRGB,
        OutSampleBayer,
    };

    enum PipelineProfile {
        BasicPipelineProfile    = 0,
        AdvancedPipelineProfile,
        ExtremePipelineProfile,
    };

    enum CaptureStage {
        BasicbayerStage,
        TonemappingStage,
    };

public:
    explicit CL3aImageProcessor ();
    virtual ~CL3aImageProcessor ();

    bool set_profile (PipelineProfile value);
    void set_stats_callback (const SmartPtr<StatsCallback> &callback);

    bool set_output_format (uint32_t fourcc);
    bool set_capture_stage (CaptureStage capture_stage);
    bool set_3a_stats_bits (uint32_t bits);

    virtual bool set_hdr (uint32_t mode);
    virtual bool set_denoise (uint32_t mode);
    virtual bool set_gamma (bool enable);
    virtual bool set_macc (bool enable);
    virtual bool set_dpc (bool enable);
    virtual bool set_tnr (uint32_t mode, uint8_t level);
    virtual bool set_tonemapping (bool enable);
    virtual bool set_newtonemapping (bool enable);
    virtual bool set_wavelet (uint32_t enable);

    PipelineProfile get_profile () const {
        return _pipeline_profile;
    }

protected:

    //derive from ImageProcessor
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);

private:
    virtual XCamReturn create_handlers ();

    bool post_config ();
    XCAM_DEAD_COPY (CL3aImageProcessor);

private:
    uint32_t                            _output_fourcc;
    uint32_t                            _3a_stats_bits;
    PipelineProfile                     _pipeline_profile;
    CaptureStage                        _capture_stage;
    SmartPtr<StatsCallback>             _stats_callback;
    SmartPtr<CLCscImageHandler>         _csc;
    SmartPtr<CLTonemappingImageHandler> _tonemapping;
    SmartPtr<CLNewTonemappingImageHandler> _newtonemapping;
    SmartPtr<CLImageScaler>             _scaler;
#if ENABLE_YEENR_HANDLER
    SmartPtr<CLEeImageHandler>          _ee;
#endif
    SmartPtr<CLWaveletDenoiseImageHandler>   _wavelet;
    SmartPtr<CLNewWaveletDenoiseImageHandler>   _newwavelet;

    // simple 3a bayer pipeline
    SmartPtr<CLBayerBasicImageHandler>  _bayer_basic_pipe;
    SmartPtr<CLBayerPipeImageHandler>   _bayer_pipe;
    SmartPtr<CLYuvPipeImageHandler>     _yuv_pipe;

    uint32_t                            _hdr_mode;
    uint32_t                            _tnr_mode;
    bool                                _enable_gamma;
    bool                                _enable_tonemapping;
    bool                                _enable_newtonemapping;
    bool                                _enable_macc;
    bool                                _enable_dpc;
    bool                                _enable_wavelet;
    bool                                _enable_newwavelet;
    uint32_t                            _snr_mode; // spatial nr mode
};

};
#endif //XCAM_CL_3A_IMAGE_PROCESSOR_H
