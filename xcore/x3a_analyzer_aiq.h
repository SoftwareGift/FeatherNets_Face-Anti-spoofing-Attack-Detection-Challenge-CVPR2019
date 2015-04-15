/*
 * x3a_analyzer_aiq.h - 3a analyzer from AIQ
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_3A_ANALYZER_AIQ_H
#define XCAM_3A_ANALYZER_AIQ_H

#include "xcam_utils.h"
#include "x3a_analyzer.h"
#include <linux/atomisp.h>

namespace XCam {

class AiqCompositor;
class IspController;

class X3aAnalyzerAiq
    : public X3aAnalyzer
{
public:
    explicit X3aAnalyzerAiq (SmartPtr<IspController> &isp, const char *cpf_path);
    ~X3aAnalyzerAiq ();

private:

    XCAM_DEAD_COPY (X3aAnalyzerAiq);

protected:
    virtual SmartPtr<AeHandler> create_ae_handler ();
    virtual SmartPtr<AwbHandler> create_awb_handler ();
    virtual SmartPtr<AfHandler> create_af_handler ();
    virtual SmartPtr<CommonHandler> create_common_handler ();

    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate);
    virtual XCamReturn internal_deinit ();

    virtual XCamReturn configure_3a ();
    virtual XCamReturn pre_3a_analyze (SmartPtr<X3aStats> &stats);
    virtual XCamReturn post_3a_analyze (X3aResultList &results);

private:
    SmartPtr <AiqCompositor>          _aiq_compositor;

    SmartPtr <IspController>          _isp;
    struct atomisp_sensor_mode_data   _sensor_mode_data;
    char                             *_cpf_path;
};

};
#endif //XCAM_3A_ANALYZER_AIQ_H
