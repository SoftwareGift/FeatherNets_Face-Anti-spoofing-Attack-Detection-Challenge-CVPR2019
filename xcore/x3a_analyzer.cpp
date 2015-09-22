/*
 * x3a_analyzer.cpp - 3a analyzer
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

#include "xcam_analyzer.h"
#include "x3a_analyzer.h"
#include "x3a_stats_pool.h"

namespace XCam {

X3aAnalyzer::X3aAnalyzer (const char *name)
    : XAnalyzer (name)
    , _ae_handler (NULL)
    , _awb_handler (NULL)
    , _af_handler (NULL)
    , _common_handler (NULL)
{
}

X3aAnalyzer::~X3aAnalyzer()
{
}

XCamReturn
X3aAnalyzer::create_handlers ()
{
    SmartPtr<AeHandler> ae_handler;
    SmartPtr<AwbHandler> awb_handler;
    SmartPtr<AfHandler> af_handler;
    SmartPtr<CommonHandler> common_handler;

    if (_ae_handler.ptr() && _awb_handler.ptr() &&
            _af_handler.ptr() && _common_handler.ptr())
        return XCAM_RETURN_NO_ERROR;

    ae_handler = create_ae_handler ();
    awb_handler = create_awb_handler ();
    af_handler = create_af_handler ();
    common_handler = create_common_handler ();

    if (!ae_handler.ptr() || !awb_handler.ptr() || !af_handler.ptr() || !common_handler.ptr()) {
        XCAM_LOG_WARNING ("create handlers failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    _ae_handler = ae_handler;
    _awb_handler = awb_handler;
    _af_handler = af_handler;
    _common_handler = common_handler;

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzer::release_handlers ()
{
    _ae_handler.release ();
    _awb_handler.release ();
    _af_handler.release ();
    _common_handler.release ();

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzer::configure ()
{
    return configure_3a ();
}

XCamReturn
X3aAnalyzer::analyze (SmartPtr<BufferProxy> &buffer)
{
    SmartPtr<X3aStats> stats = buffer.dynamic_cast_ptr<X3aStats> ();

    return analyze_3a_statistics (stats);
}

XCamReturn
X3aAnalyzer::push_3a_stats (const SmartPtr<X3aStats> &stats)
{
    return XAnalyzer::push_buffer (stats);
}


XCamReturn
X3aAnalyzer::analyze_3a_statistics (SmartPtr<X3aStats> &stats)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    X3aResultList results;

    ret = pre_3a_analyze (stats);
    if (ret != XCAM_RETURN_NO_ERROR) {
        notify_calculation_failed(
            NULL, stats->get_timestamp (), "pre 3a analyze failed");
        return ret;
    }

    ret = _ae_handler->analyze (results);
    if (ret != XCAM_RETURN_NO_ERROR) {
        notify_calculation_failed(
            _ae_handler.ptr(), stats->get_timestamp (), "ae calculation failed");
        return ret;
    }

    ret = _awb_handler->analyze (results);
    if (ret != XCAM_RETURN_NO_ERROR) {
        notify_calculation_failed(
            _awb_handler.ptr(), stats->get_timestamp (), "awb calculation failed");
        return ret;
    }

    ret = _af_handler->analyze (results);
    if (ret != XCAM_RETURN_NO_ERROR) {
        notify_calculation_failed(
            _af_handler.ptr(), stats->get_timestamp (), "af calculation failed");
        return ret;
    }

    ret = _common_handler->analyze (results);
    if (ret != XCAM_RETURN_NO_ERROR) {
        notify_calculation_failed(
            _common_handler.ptr(), stats->get_timestamp (), "3a other calculation failed");
        return ret;
    }

    ret = post_3a_analyze (results);
    if (ret != XCAM_RETURN_NO_ERROR) {
        notify_calculation_failed(
            NULL, stats->get_timestamp (), "3a collect results failed");
        return ret;
    }

    if (!results.empty ()) {
        set_results_timestamp(results, stats->get_timestamp ());
        notify_calculation_done (results);
    }

    return ret;
}

/* AWB */
bool
X3aAnalyzer::set_awb_mode (XCamAwbMode mode)
{
    XCAM_ASSERT (_awb_handler.ptr());
    return _awb_handler->set_mode (mode);
}

bool
X3aAnalyzer::set_awb_speed (double speed)
{
    XCAM_ASSERT (_awb_handler.ptr());
    return _awb_handler->set_speed (speed);
}

bool
X3aAnalyzer::set_awb_color_temperature_range (uint32_t cct_min, uint32_t cct_max)
{
    XCAM_ASSERT (_awb_handler.ptr());
    return _awb_handler->set_color_temperature_range (cct_min, cct_max);
}

bool
X3aAnalyzer::set_awb_manual_gain (double gr, double r, double b, double gb)
{
    XCAM_ASSERT (_awb_handler.ptr());
    return _awb_handler->set_manual_gain (gr, r, b, gb);
}

/* AE */
bool
X3aAnalyzer::set_ae_mode (XCamAeMode mode)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_mode (mode);
}

bool
X3aAnalyzer::set_ae_metering_mode (XCamAeMeteringMode mode)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_metering_mode (mode);
}

bool
X3aAnalyzer::set_ae_window (XCam3AWindow *window, uint8_t count)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_window (window, count);
}

bool
X3aAnalyzer::set_ae_ev_shift (double ev_shift)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_ev_shift (ev_shift);
}

bool
X3aAnalyzer::set_ae_speed (double speed)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_speed (speed);
}

bool
X3aAnalyzer::set_ae_flicker_mode (XCamFlickerMode flicker)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_flicker_mode (flicker);
}

XCamFlickerMode
X3aAnalyzer::get_ae_flicker_mode ()
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->get_flicker_mode ();
}

uint64_t
X3aAnalyzer::get_ae_current_exposure_time ()
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->get_current_exposure_time();
}

double
X3aAnalyzer::get_ae_current_analog_gain ()
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->get_current_analog_gain ();
}

bool
X3aAnalyzer::set_ae_manual_exposure_time (int64_t time_in_us)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_manual_exposure_time (time_in_us);
}

bool
X3aAnalyzer::set_ae_manual_analog_gain (double gain)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_manual_analog_gain (gain);
}

bool
X3aAnalyzer::set_ae_aperture (double fn)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_aperture (fn);
}

bool
X3aAnalyzer::set_ae_max_analog_gain (double max_gain)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_max_analog_gain (max_gain);
}

double
X3aAnalyzer::get_ae_max_analog_gain ()
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->get_max_analog_gain();
}

bool
X3aAnalyzer::set_ae_exposure_time_range (int64_t min_time_in_us, int64_t max_time_in_us)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->set_exposure_time_range (min_time_in_us, max_time_in_us);
}

bool
X3aAnalyzer::get_ae_exposure_time_range (int64_t *min_time_in_us, int64_t *max_time_in_us)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->get_exposure_time_range (min_time_in_us, max_time_in_us);
}

/* DVS */
bool
X3aAnalyzer::set_dvs (bool enable)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_dvs (enable);
}

bool
X3aAnalyzer::set_gbce (bool enable)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_gbce (enable);
}

bool
X3aAnalyzer::set_night_mode (bool enable)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_night_mode (enable);
}

bool
X3aAnalyzer::set_color_effect (XCamColorEffect type)
{

    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_color_effect (type);
}

/* Picture quality */
bool
X3aAnalyzer::set_noise_reduction_level (double level)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_noise_reduction_level (level);
}

bool
X3aAnalyzer::set_temporal_noise_reduction_level (double level)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_temporal_noise_reduction_level (level);
}

bool
X3aAnalyzer::set_manual_brightness (double level)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_manual_brightness (level);
}

bool
X3aAnalyzer::set_manual_contrast (double level)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_manual_contrast (level);
}

bool
X3aAnalyzer::set_manual_hue (double level)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_manual_hue (level);
}

bool
X3aAnalyzer::set_manual_saturation (double level)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_manual_saturation (level);
}

bool
X3aAnalyzer::set_manual_sharpness (double level)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_manual_sharpness (level);
}

bool
X3aAnalyzer::set_gamma_table (double *r_table, double *g_table, double *b_table)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->set_gamma_table (r_table, g_table, b_table);
}

bool
X3aAnalyzer::set_parameter_brightness(double level)
{
    _brightness_level_param = level;
    return true;
}

bool
X3aAnalyzer::update_awb_parameters (const XCamAwbParam &params)
{
    XCAM_ASSERT (_awb_handler.ptr());
    return _awb_handler->update_parameters (params);
}

bool
X3aAnalyzer::update_common_parameters (const XCamCommonParam &params)
{
    XCAM_ASSERT (_common_handler.ptr());
    return _common_handler->update_parameters (params);
}

bool
X3aAnalyzer::update_ae_parameters (const XCamAeParam &params)
{
    XCAM_ASSERT (_ae_handler.ptr());
    return _ae_handler->update_parameters (params);
}

bool
X3aAnalyzer::update_af_parameters (const XCamAfParam &params)
{
    XCAM_ASSERT (_af_handler.ptr());
    return _af_handler->update_parameters (params);
}

};
