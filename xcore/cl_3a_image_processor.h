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
#include "cl_image_processor.h"
#include "stats_callback_interface.h"

namespace XCam {

class CLBayer2RGBImageHandler;
class CLCscImageHandler;
class CLGammaImageHandler;
class CL3AStatsCalculator;
class CLWbImageHandler;

class CL3aImageProcessor
    : public CLImageProcessor
{
    enum OutSampleType {
        OutSampleYuv,
        OutSampleRGB,
        OutSampleBayer,
    };

public:
    explicit CL3aImageProcessor ();
    virtual ~CL3aImageProcessor ();

    void set_stats_callback (const SmartPtr<StatsCallback> &callback);

    bool set_output_format (uint32_t fourcc);
    void set_hdr (bool enable) {
        _enable_hdr = enable;
    }
    void set_denoise (bool enable) {
        _enable_denoise = enable;
    }
    void set_gamma (bool enable) {
        _enable_gamma = enable;
    }
    void set_snr (bool enable) {
        _enable_snr = enable;
    }

protected:

    //derive from ImageProcessor
    virtual bool can_process_result (SmartPtr<X3aResult> &result);
    virtual XCamReturn apply_3a_results (X3aResultList &results);
    virtual XCamReturn apply_3a_result (SmartPtr<X3aResult> &result);

private:
    virtual XCamReturn create_handlers ();
    XCAM_DEAD_COPY (CL3aImageProcessor);

private:
    uint32_t                           _output_fourcc;
    bool                               _enable_hdr;
    bool                               _enable_denoise;
    bool                               _enable_gamma;
    OutSampleType                      _out_smaple_type;
    bool                               _enable_snr;

    SmartPtr<StatsCallback>            _stats_callback;

    SmartPtr<CLImageHandler>           _black_level;
    SmartPtr<CLBayer2RGBImageHandler>  _demosaic;
    SmartPtr<CLImageHandler>           _hdr;
    SmartPtr<CLCscImageHandler>        _csc;
    SmartPtr<CLImageHandler>           _denoise;
    SmartPtr<CLGammaImageHandler>      _gamma;
    SmartPtr<CL3AStatsCalculator>      _x3a_stats_calculator;
    SmartPtr<CLWbImageHandler>         _wb;
    SmartPtr<CLImageHandler>           _snr;
};

};
#endif //XCAM_CL_3A_IMAGE_PROCESSOR_H
