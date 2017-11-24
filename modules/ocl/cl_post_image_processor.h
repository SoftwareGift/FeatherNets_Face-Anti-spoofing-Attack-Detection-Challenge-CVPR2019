/*
 * cl_post_image_processor.h - CL post image processor
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_CL_POST_IMAGE_PROCESSOR_H
#define XCAM_CL_POST_IMAGE_PROCESSOR_H

#include <xcam_std.h>
#include <base/xcam_3a_types.h>
#include <ocl/cl_image_processor.h>
#include <stats_callback_interface.h>
#include <ocl/cl_blender.h>
#include <ocl/cl_utils.h>

namespace XCam {

class CLTnrImageHandler;
class CLRetinexImageHandler;
class CLCscImageHandler;
class CLDefogDcpImageHandler;
class CLWaveletDenoiseImageHandler;
class CLNewWaveletDenoiseImageHandler;
class CL3DDenoiseImageHandler;
class CLImageScaler;
class CLWireFrameImageHandler;
class CLImageWarpHandler;
class CLImage360Stitch;
class CLVideoStabilizer;

class CLPostImageProcessor
    : public CLImageProcessor
{
public:
    enum OutSampleType {
        OutSampleYuv,
        OutSampleRGB,
        OutSampleBayer,
    };

    enum CLTnrMode {
        TnrDisable = 0,
        TnrYuv,
    };

    enum CLDefogMode {
        DefogDisabled = 0,
        DefogRetinex,
        DefogDarkChannelPrior,
    };

    enum CL3DDenoiseMode {
        Denoise3DDisabled = 0,
        Denoise3DYuv,
        Denoise3DUV,
    };

public:
    explicit CLPostImageProcessor ();
    virtual ~CLPostImageProcessor ();

    bool set_output_format (uint32_t fourcc);
    void set_stats_callback (const SmartPtr<StatsCallback> &callback);

    bool set_scaler_factor (const double factor);
    double get_scaler_factor () const {
        return _scaler_factor;
    }
    bool is_scaled () {
        return _enable_scaler;
    }

    virtual bool set_tnr (CLTnrMode mode);
    virtual bool set_defog_mode (CLDefogMode mode);
    virtual bool set_wavelet (CLWaveletBasis basis, uint32_t channel, bool bayes_shrink);
    virtual bool set_3ddenoise_mode (CL3DDenoiseMode mode, uint8_t ref_frame_count);
    virtual bool set_scaler (bool enable);
    virtual bool set_wireframe (bool enable);
    virtual bool set_image_warp (bool enable);
    virtual bool set_image_stitch (
        bool enable_stitch, bool enable_seam, CLBlenderScaleMode scale_mode, bool enable_fisheye_map,
        bool lsc, bool fm_ocl, uint32_t stitch_width, uint32_t stitch_height, uint32_t res_mode);

protected:
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);

private:
    virtual XCamReturn create_handlers ();

    XCAM_DEAD_COPY (CLPostImageProcessor);

private:
    uint32_t                                  _output_fourcc;
    OutSampleType                             _out_sample_type;
    SmartPtr<StatsCallback>                   _stats_callback;

    SmartPtr<CLTnrImageHandler>               _tnr;
    SmartPtr<CLRetinexImageHandler>           _retinex;
    SmartPtr<CLDefogDcpImageHandler>          _defog_dcp;
    SmartPtr<CLWaveletDenoiseImageHandler>    _wavelet;
    SmartPtr<CLNewWaveletDenoiseImageHandler> _newwavelet;
    SmartPtr<CL3DDenoiseImageHandler>         _3d_denoise;
    SmartPtr<CLImageScaler>                   _scaler;
    SmartPtr<CLWireFrameImageHandler>         _wireframe;
    SmartPtr<CLCscImageHandler>               _csc;
    SmartPtr<CLImageWarpHandler>              _image_warp;
    SmartPtr<CLImage360Stitch>                _stitch;
    SmartPtr<CLVideoStabilizer>               _video_stab;

    double                                    _scaler_factor;

    CLTnrMode                                 _tnr_mode;
    CLDefogMode                               _defog_mode;
    CLWaveletBasis                            _wavelet_basis;
    uint32_t                                  _wavelet_channel;
    bool                                      _wavelet_bayes_shrink;
    CL3DDenoiseMode                           _3d_denoise_mode;
    uint8_t                                   _3d_denoise_ref_count;
    bool                                      _enable_scaler;
    bool                                      _enable_wireframe;
    bool                                      _enable_image_warp;
    bool                                      _enable_stitch;
    bool                                      _stitch_enable_seam;
    bool                                      _stitch_fisheye_map;
    bool                                      _stitch_lsc;
    bool                                      _stitch_fm_ocl;
    CLBlenderScaleMode                        _stitch_scale_mode;
    uint32_t                                  _stitch_width;
    uint32_t                                  _stitch_height;
    uint32_t                                  _stitch_res_mode;
    uint32_t                                  _surround_mode;
};

};
#endif // XCAM_CL_POST_IMAGE_PROCESSOR_H
