/*
 * x3a_analyzer_simple.cpp - a simple 3a analyzer
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

#include "x3a_analyzer_simple.h"

namespace XCam {

#define SIMPLE_MIN_TARGET_EXPOSURE_TIME  5000 //5ms
#define SIMPLE_MAX_TARGET_EXPOSURE_TIME  33000 //33ms
#define SIMPLE_DEFAULT_BLACK_LEVEL       0.05

class SimpleAeHandler
    : public AeHandler
{
public:
    SimpleAeHandler (X3aAnalyzerSimple *analyzer)
        : _analyzer (analyzer)
    {}
    ~SimpleAeHandler () {}

    virtual XCamReturn analyze (X3aResultList &output) {
        return _analyzer->analyze_ae (output);
    }
private:
    X3aAnalyzerSimple *_analyzer;
};

class SimpleAwbHandler
    : public AwbHandler
{
public:
    SimpleAwbHandler (X3aAnalyzerSimple *analyzer)
        : _analyzer (analyzer)
    {}
    ~SimpleAwbHandler () {}

    virtual XCamReturn analyze (X3aResultList &output) {
        return _analyzer->analyze_awb (output);
    }
private:
    X3aAnalyzerSimple *_analyzer;

};

class SimpleAfHandler
    : public AfHandler
{
public:
    SimpleAfHandler (X3aAnalyzerSimple *analyzer)
        : _analyzer (analyzer)
    {}
    ~SimpleAfHandler () {}

    virtual XCamReturn analyze (X3aResultList &output) {
        return _analyzer->analyze_af (output);
    }

private:
    X3aAnalyzerSimple *_analyzer;
};

class SimpleCommonHandler
    : public CommonHandler
{
public:
    SimpleCommonHandler (X3aAnalyzerSimple *analyzer)
        : _analyzer (analyzer)
    {}
    ~SimpleCommonHandler () {}

    virtual XCamReturn analyze (X3aResultList &output) {
        XCAM_UNUSED (output);
        return XCAM_RETURN_NO_ERROR;
    }

private:
    X3aAnalyzerSimple *_analyzer;
};

X3aAnalyzerSimple::X3aAnalyzerSimple ()
    : X3aAnalyzer ("X3aAnalyzerSimple")
    , _last_target_exposure ((double)SIMPLE_MIN_TARGET_EXPOSURE_TIME)
    , _is_ae_started (false)
    , _ae_calculation_interval (0)
{
}

X3aAnalyzerSimple::~X3aAnalyzerSimple ()
{
}

SmartPtr<AeHandler>
X3aAnalyzerSimple::create_ae_handler ()
{
    SimpleAeHandler *handler = new SimpleAeHandler (this);
    return handler;
}

SmartPtr<AwbHandler>
X3aAnalyzerSimple::create_awb_handler ()
{
    SimpleAwbHandler *handler = new SimpleAwbHandler (this);
    return handler;
}

SmartPtr<AfHandler>
X3aAnalyzerSimple::create_af_handler ()
{
    SimpleAfHandler *handler = new SimpleAfHandler (this);
    return handler;
}

SmartPtr<CommonHandler>
X3aAnalyzerSimple::create_common_handler ()
{
    SimpleCommonHandler *handler = new SimpleCommonHandler (this);
    return handler;
}

XCamReturn
X3aAnalyzerSimple::configure_3a ()
{
    _is_ae_started = false;
    _ae_calculation_interval = 0;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerSimple::pre_3a_analyze (SmartPtr<X3aStats> &stats)
{
    _current_stats = stats;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerSimple::post_3a_analyze (X3aResultList &results)
{
    _current_stats.release ();

    XCam3aResultBlackLevel black_level;
    SmartPtr<X3aBlackLevelResult> bl_result = new X3aBlackLevelResult (XCAM_3A_RESULT_BLACK_LEVEL);

    xcam_mem_clear (black_level);
    black_level.r_level = SIMPLE_DEFAULT_BLACK_LEVEL;
    black_level.gr_level = SIMPLE_DEFAULT_BLACK_LEVEL;
    black_level.gb_level = SIMPLE_DEFAULT_BLACK_LEVEL;
    black_level.b_level = SIMPLE_DEFAULT_BLACK_LEVEL;
    bl_result->set_standard_result (black_level);
    results.push_back (bl_result);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerSimple::analyze_awb (X3aResultList &output)
{
    const XCam3AStats *stats = _current_stats->get_stats ();
    double sum_r = 0.0, sum_gr = 0.0, sum_gb = 0.0, sum_b = 0.0;
    double avg_r = 0.0, avg_gr = 0.0, avg_gb = 0.0, avg_b = 0.0;
    double target_avg = 0.0;
    XCam3aResultWhiteBalance wb;

    xcam_mem_clear (wb);
    XCAM_ASSERT (stats);

    // calculate avg r, gr, gb, b
    for (uint32_t i = 0; i < stats->info.height; ++i)
        for (uint32_t j = 0; j < stats->info.width; ++j) {
            sum_r += (double)(stats->stats[i * stats->info.aligned_width + j].avg_r);
            sum_gr += (double)(stats->stats[i * stats->info.aligned_width + j].avg_gr);
            sum_gb += (double)(stats->stats[i * stats->info.aligned_width + j].avg_gb);
            sum_b += (double)(stats->stats[i * stats->info.aligned_width + j].avg_b);
        }

    avg_r = sum_r / (stats->info.width * stats->info.height);
    avg_gr = sum_gr / (stats->info.width * stats->info.height);
    avg_gb = sum_gb / (stats->info.width * stats->info.height);
    avg_b = sum_b / (stats->info.width * stats->info.height);

    target_avg =  (avg_gr + avg_gb) / 2;
    wb.r_gain = target_avg / avg_r;
    wb.b_gain = target_avg / avg_b;
    wb.gr_gain = 1.0;
    wb.gb_gain = 1.0;

    SmartPtr<X3aWhiteBalanceResult> result = new X3aWhiteBalanceResult (XCAM_3A_RESULT_WHITE_BALANCE);
    result->set_standard_result (wb);
    output.push_back (result);

    XCAM_LOG_DEBUG ("X3aAnalyzerSimple analyze awb, r:%f, gr:%f, gb:%f, b:%f",
                    wb.r_gain, wb.gr_gain, wb.gb_gain, wb.b_gain);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerSimple::analyze_ae (X3aResultList &output)
{
    static const uint32_t expect_y_mean = 110;

    const XCam3AStats *stats = _current_stats->get_stats ();
    XCAM_FAIL_RETURN(
        WARNING,
        stats,
        XCAM_RETURN_ERROR_UNKNOWN,
        "failed to get XCam3AStats");

    double sum_y = 0.0;
    double target_exposure = 1.0;
    SmartPtr<X3aExposureResult> result = new X3aExposureResult (XCAM_3A_RESULT_EXPOSURE);
    XCam3aResultExposure exposure;

    xcam_mem_clear (exposure);
    exposure.digital_gain = 1.0;

    if (!_is_ae_started) {
        _last_target_exposure = SIMPLE_MIN_TARGET_EXPOSURE_TIME;
        exposure.exposure_time = _last_target_exposure;
        exposure.analog_gain = 1.0;

        result->set_standard_result (exposure);
        output.push_back (result);
        _is_ae_started = true;
        return XCAM_RETURN_NO_ERROR;
    }

    if (_ae_calculation_interval % 10 == 0) {
        for (uint32_t i = 0; i < stats->info.height; ++i)
            for (uint32_t j = 0; j < stats->info.width; ++j) {
                sum_y += (double)(stats->stats[i * stats->info.aligned_width + j].avg_y);
            }
        sum_y /= (stats->info.width * stats->info.height);
        target_exposure = (expect_y_mean / sum_y) * _last_target_exposure;
        target_exposure = XCAM_MAX (target_exposure, SIMPLE_MIN_TARGET_EXPOSURE_TIME);

        if (target_exposure > SIMPLE_MAX_TARGET_EXPOSURE_TIME * 255)
            target_exposure = SIMPLE_MAX_TARGET_EXPOSURE_TIME * 255;

        if (target_exposure > SIMPLE_MAX_TARGET_EXPOSURE_TIME) {
            exposure.exposure_time = SIMPLE_MAX_TARGET_EXPOSURE_TIME;
            exposure.analog_gain = target_exposure / exposure.exposure_time;
        } else {
            exposure.exposure_time = target_exposure;
            exposure.analog_gain = 1.0;
        }

        result->set_standard_result (exposure);
        output.push_back (result);
        _last_target_exposure = target_exposure;
    }

    _ae_calculation_interval++;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn X3aAnalyzerSimple::analyze_af (X3aResultList &output)
{
    XCAM_UNUSED (output);
    return XCAM_RETURN_NO_ERROR;
}

};
