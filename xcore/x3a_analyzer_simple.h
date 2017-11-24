/*
 * x3a_analyzer_simple.h - a simple 3a analyzer
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

#ifndef XCAM_3A_ANALYZER_SIMPLE_H
#define XCAM_3A_ANALYZER_SIMPLE_H

#include <xcam_std.h>
#include <x3a_analyzer.h>
#include <x3a_stats_pool.h>

namespace XCam {

class X3aAnalyzerSimple
    : public X3aAnalyzer
{
public:
    explicit X3aAnalyzerSimple ();
    ~X3aAnalyzerSimple ();

private:

    XCAM_DEAD_COPY (X3aAnalyzerSimple);

protected:
    virtual SmartPtr<AeHandler> create_ae_handler ();
    virtual SmartPtr<AwbHandler> create_awb_handler ();
    virtual SmartPtr<AfHandler> create_af_handler ();
    virtual SmartPtr<CommonHandler> create_common_handler ();

    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate) {
        XCAM_UNUSED (width);
        XCAM_UNUSED (height);
        XCAM_UNUSED (framerate);
        return XCAM_RETURN_NO_ERROR;
    }
    virtual XCamReturn internal_deinit () {
        _is_ae_started = false;
        _ae_calculation_interval = 0;
        return XCAM_RETURN_NO_ERROR;
    }
    virtual XCamReturn configure_3a ();
    virtual XCamReturn pre_3a_analyze (SmartPtr<X3aStats> &stats);
    virtual XCamReturn post_3a_analyze (X3aResultList &results);

public:
    XCamReturn analyze_ae (X3aResultList &output);
    XCamReturn analyze_awb (X3aResultList &output);
    XCamReturn analyze_af (X3aResultList &output);

private:
    SmartPtr<X3aStats>                _current_stats;
    double                            _last_target_exposure;
    bool                              _is_ae_started;
    uint32_t                          _ae_calculation_interval;
};

};
#endif //XCAM_3A_ANALYZER_SIMPLE_H

