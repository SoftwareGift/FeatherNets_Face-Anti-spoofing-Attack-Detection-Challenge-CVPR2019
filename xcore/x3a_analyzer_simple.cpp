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
#include "x3a_statistics_queue.h"
#include <linux/atomisp.h>

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
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzerSimple::pre_3a_analyze (SmartPtr<X3aIspStatistics> &stats)
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

    xcam_mem_clear (&black_level);
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
    const struct atomisp_3a_statistics *stats = _current_stats->get_3a_stats ();
    uint32_t cell_count = stats->grid_info.bqs_per_grid_cell * stats->grid_info.bqs_per_grid_cell;
    uint32_t bits_depth = stats->grid_info.elem_bit_depth;
    double sum_r = 0.0, sum_gr = 0.0, sum_gb = 0.0, sum_b = 0.0;
    double avg_r = 0.0, avg_gr = 0.0, avg_gb = 0.0, avg_b = 0.0;
    double target_avg = 0.0;
    XCam3aResultWhiteBalance wb;

    xcam_mem_clear (&wb);

    // calculate avg r, gr, gb, b
    for (uint32_t i = 0; i < stats->grid_info.height; ++i)
        for (uint32_t j = 0; j < stats->grid_info.width; ++j) {
            sum_r += ((double)(stats->data[i * stats->grid_info.width + j].awb_r)) / cell_count;
            sum_gr += ((double)(stats->data[i * stats->grid_info.width + j].awb_gr)) / cell_count;
            sum_gb += ((double)(stats->data[i * stats->grid_info.width + j].awb_gb)) / cell_count;
            sum_b += ((double)(stats->data[i * stats->grid_info.width + j].awb_b)) / cell_count;
        }

    avg_r = sum_r / (stats->grid_info.width * stats->grid_info.height);
    avg_gr = sum_gr / (stats->grid_info.width * stats->grid_info.height);
    avg_gb = sum_gb / (stats->grid_info.width * stats->grid_info.height);
    avg_b = sum_b / (stats->grid_info.width * stats->grid_info.height);
    avg_r = avg_r / (1 << (bits_depth - 8));
    avg_gr = avg_gr / (1 << (bits_depth - 8));
    avg_gb = avg_gb / (1 << (bits_depth - 8));
    avg_b = avg_b / (1 << (bits_depth - 8));
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
    static const uint32_t expect_y_mean = 150;

    const struct atomisp_3a_statistics *stats = _current_stats->get_3a_stats ();
    uint32_t cell_count = stats->grid_info.bqs_per_grid_cell * stats->grid_info.bqs_per_grid_cell;
    uint32_t bits_depth = stats->grid_info.elem_bit_depth;
    double sum_y = 0.0;
    double target_exposure = 1.0;
    SmartPtr<X3aExposureResult> result = new X3aExposureResult (XCAM_3A_RESULT_EXPOSURE);;
    XCam3aResultExposure exposure;

    xcam_mem_clear (&exposure);
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

    for (uint32_t i = 0; i < stats->grid_info.height; ++i)
        for (uint32_t j = 0; j < stats->grid_info.width; ++j) {
            sum_y += ((double)(stats->data[i * stats->grid_info.width + j].ae_y)) / cell_count;
        }
    sum_y /= (stats->grid_info.width * stats->grid_info.height);
    sum_y /= (1 << (bits_depth - 8)); // make it in 8 bits
    target_exposure = (expect_y_mean / sum_y) * _last_target_exposure;
    target_exposure = XCAM_MAX (target_exposure, SIMPLE_MIN_TARGET_EXPOSURE_TIME);

    if (target_exposure > SIMPLE_MAX_TARGET_EXPOSURE_TIME) {
        exposure.exposure_time = SIMPLE_MAX_TARGET_EXPOSURE_TIME;
        exposure.analog_gain = target_exposure / exposure.exposure_time;
    } else {
        exposure.exposure_time = target_exposure;
        exposure.analog_gain = 1.0;
    }
    result->set_standard_result (exposure);
    output.push_back (result);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn X3aAnalyzerSimple::analyze_af (X3aResultList &output)
{
    XCAM_UNUSED (output);
    return XCAM_RETURN_NO_ERROR;
}

};
