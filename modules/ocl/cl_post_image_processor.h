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

#include "xcam_utils.h"
#include <base/xcam_3a_types.h>
#include "cl_image_processor.h"

namespace XCam {

class CLTnrImageHandler;
class CLRetinexImageHandler;
class CLCscImageHandler;
class CLDefogDcpImageHandler;
class CL3DDenoiseImageHandler;

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

    virtual bool set_tnr (CLTnrMode mode);
    virtual bool set_defog_mode (CLDefogMode mode);
    virtual bool set_3ddenoise_mode (CL3DDenoiseMode mode, uint8_t ref_frame_count);

protected:
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);

private:
    virtual XCamReturn create_handlers ();

    XCAM_DEAD_COPY (CLPostImageProcessor);

private:
    uint32_t                               _output_fourcc;
    OutSampleType                          _out_sample_type;

    SmartPtr<CLTnrImageHandler>            _tnr;
    SmartPtr<CLRetinexImageHandler>        _retinex;
    SmartPtr<CLDefogDcpImageHandler>       _defog_dcp;
    SmartPtr<CLCscImageHandler>            _csc;
    SmartPtr<CL3DDenoiseImageHandler>      _3d_denoise;

    CLTnrMode                              _tnr_mode;
    CLDefogMode                            _defog_mode;
    CL3DDenoiseMode                        _3d_denoise_mode;
    uint8_t                                _3d_denoise_ref_count;
};

};
#endif // XCAM_CL_POST_IMAGE_PROCESSOR_H
