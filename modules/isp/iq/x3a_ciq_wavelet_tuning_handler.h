/*
 * x3a_ciq_wavelet_tuning_handler.h - x3a Common IQ Wavelet denoise tuning handler
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_3A_CIQ_WAVELET_TUNING_HANDLER_H
#define XCAM_3A_CIQ_WAVELET_TUNING_HANDLER_H

#include "xcam_utils.h"

namespace XCam {

class X3aCiqWaveletTuningHandler
    : public X3aCiqTuningHandler
{
public:
    explicit X3aCiqWaveletTuningHandler ();
    virtual ~X3aCiqWaveletTuningHandler ();

    virtual XCamReturn analyze (X3aResultList &output);

private:
    XCAM_DEAD_COPY (X3aCiqWaveletTuningHandler);

};

};

#endif // XCAM_3A_CIQ_WAVELET_TUNING_HANDLER_H
