/*
 * handler_interface.cpp - handler interface
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

#include "handler_interface.h"

namespace XCam {

AeHandler::AeHandler()
{
    reset_parameters ();
}

void
AeHandler::reset_parameters ()
{
    // in case missing any parameters
    xcam_mem_clear (_params);

    _params.mode = XCAM_AE_MODE_AUTO;
    _params.metering_mode = XCAM_AE_METERING_MODE_AUTO;
    _params.flicker_mode = XCAM_AE_FLICKER_MODE_AUTO;
    _params.speed = 1.0;
    _params.exposure_time_min = UINT64_C(0);
    _params.exposure_time_max = UINT64_C(0);
    _params.max_analog_gain = 0.0;
    _params.manual_exposure_time = UINT64_C (0);
    _params.manual_analog_gain = 0.0;
    _params.aperture_fn = 0.0;
    _params.ev_shift = 0.0;

    _params.window.x_start = 0;
    _params.window.y_start = 0;
    _params.window.x_end = 0;
    _params.window.y_end = 0;
    _params.window.weight = 0;

    xcam_mem_clear (_params.window_list);
}

bool
AeHandler::set_mode (XCamAeMode mode)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.mode = mode;

    XCAM_LOG_DEBUG ("ae set mode [%d]", mode);
    return true;
}

bool
AeHandler::set_metering_mode (XCamAeMeteringMode mode)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.metering_mode = mode;

    XCAM_LOG_DEBUG ("ae set metering mode [%d]", mode);
    return true;
}

bool
AeHandler::set_window (XCam3AWindow *window)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.window = *window;

    XCAM_LOG_DEBUG ("ae set metering mode window [x:%d, y:%d, x_end:%d, y_end:%d, weight:%d]",
                    window->x_start,
                    window->y_start,
                    window->x_end,
                    window->y_end,
                    window->weight);
    return true;
}

bool
AeHandler::set_window (XCam3AWindow *window, uint8_t count)
{
    if (0 == count) {
        XCAM_LOG_WARNING ("invalid input parameter, window count = %d, reset to default value", count);
        XCam3AWindow defaultWindow = {0, 0, 1000, 1000, 15};
        set_window(&defaultWindow);
        _params.window_list[0] = defaultWindow;
        return true;
    }

    if (XCAM_AE_MAX_METERING_WINDOW_COUNT < count) {
        XCAM_LOG_WARNING ("invalid input parameter, window count = %d, reset count to maximum", count);
        count = XCAM_AE_MAX_METERING_WINDOW_COUNT;
    }

    AnalyzerHandler::HandlerLock lock(this);

    _params.window = *window;

    for (int i = 0; i < count; i++) {
        XCAM_LOG_DEBUG ("window start point(%d, %d), end point(%d, %d), weight = %d",
                        window[i].x_start, window[i].y_start, window[i].x_end, window[i].y_end, window[i].weight);

        _params.window_list[i] = window[i];
        if (_params.window.weight < window[i].weight) {
            _params.window.weight = window[i].weight;
            _params.window.x_start = window[i].x_start;
            _params.window.y_start = window[i].y_start;
            _params.window.x_end = window[i].x_end;
            _params.window.y_end = window[i].y_end;
        }
    }

    XCAM_LOG_DEBUG ("ae set metering mode window [x:%d, y:%d, x_end:%d, y_end:%d, weight:%d]",
                    _params.window.x_start,
                    _params.window.y_start,
                    _params.window.x_end,
                    _params.window.y_end,
                    _params.window.weight);

    return true;
}

bool
AeHandler::set_ev_shift (double ev_shift)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.ev_shift = ev_shift;

    XCAM_LOG_DEBUG ("ae set ev shift:%.03f", ev_shift);
    return true;
}

bool
AeHandler::set_speed (double speed)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.speed = speed;

    XCAM_LOG_DEBUG ("ae set speed:%.03f", speed);
    return true;
}

bool
AeHandler::set_flicker_mode (XCamFlickerMode flicker)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.flicker_mode = flicker;

    XCAM_LOG_DEBUG ("ae set flicker:%d", flicker);
    return true;
}

XCamFlickerMode
AeHandler::get_flicker_mode ()
{
    AnalyzerHandler::HandlerLock lock(this);
    return _params.flicker_mode;
}

int64_t
AeHandler::get_current_exposure_time ()
{
    AnalyzerHandler::HandlerLock lock(this);
    if (_params.mode == XCAM_AE_MODE_MANUAL)
        return _params.manual_exposure_time;
    return INT64_C(-1);
}

double
AeHandler::get_current_analog_gain ()
{
    AnalyzerHandler::HandlerLock lock(this);
    if (_params.mode == XCAM_AE_MODE_MANUAL)
        return _params.manual_analog_gain;
    return 0.0;
}

bool
AeHandler::set_manual_exposure_time (int64_t time_in_us)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.manual_exposure_time = time_in_us;

    XCAM_LOG_DEBUG ("ae set manual exposure time: %" PRId64 "us", time_in_us);
    return true;
}

bool
AeHandler::set_manual_analog_gain (double gain)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.manual_analog_gain = gain;

    XCAM_LOG_DEBUG ("ae set manual analog gain: %.03f", gain);
    return true;
}

bool
AeHandler::set_aperture (double fn)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.aperture_fn = fn;

    XCAM_LOG_DEBUG ("ae set aperture fn: %.03f", fn);
    return true;
}

bool
AeHandler::set_max_analog_gain (double max_gain)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.max_analog_gain = max_gain;

    XCAM_LOG_DEBUG ("ae set max analog_gain: %.03f", max_gain);
    return true;
}

double AeHandler::get_max_analog_gain ()
{
    AnalyzerHandler::HandlerLock lock(this);
    return _params.max_analog_gain;
}

bool AeHandler::set_exposure_time_range (int64_t min_time_in_us, int64_t max_time_in_us)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.exposure_time_min = min_time_in_us;
    _params.exposure_time_max = max_time_in_us;

    XCAM_LOG_DEBUG ("ae set exposrue range[%" PRId64 "us, %" PRId64 "us]", min_time_in_us, max_time_in_us);
    return true;
}

bool
AeHandler::update_parameters (const XCamAeParam &params)
{
    {
        AnalyzerHandler::HandlerLock lock (this);
        _params = params;
    }
    XCAM_LOG_DEBUG ("ae parameters updated");
    return true;
}

bool
AeHandler::get_exposure_time_range (int64_t *min_time_in_us, int64_t *max_time_in_us)
{
    XCAM_ASSERT (min_time_in_us && max_time_in_us);

    AnalyzerHandler::HandlerLock lock(this);
    *min_time_in_us = _params.exposure_time_min;
    *max_time_in_us = _params.exposure_time_max;

    return true;
}

AwbHandler::AwbHandler()
{
    reset_parameters ();
}

void
AwbHandler::reset_parameters ()
{
    xcam_mem_clear (_params);
    _params.mode = XCAM_AWB_MODE_AUTO;
    _params.speed = 1.0;
    _params.cct_min = 0;
    _params.cct_max = 0;
    _params.gr_gain = 0.0;
    _params.r_gain = 0.0;
    _params.b_gain = 0.0;
    _params.gb_gain = 0.0;

    _params.window.x_start = 0;
    _params.window.y_start = 0;
    _params.window.x_end = 0;
    _params.window.y_end = 0;
    _params.window.weight = 0;
}

bool
AwbHandler::set_mode (XCamAwbMode mode)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.mode = mode;

    XCAM_LOG_DEBUG ("awb set mode [%d]", mode);
    return true;
}

bool
AwbHandler::set_speed (double speed)
{
    XCAM_FAIL_RETURN (
        ERROR,
        (0.0 < speed) && (speed <= 1.0),
        false,
        "awb speed(%f) is out of range, suggest (0.0, 1.0]", speed);

    AnalyzerHandler::HandlerLock lock(this);
    _params.speed = speed;

    XCAM_LOG_DEBUG ("awb set speed [%f]", speed);
    return true;
}

bool
AwbHandler::set_color_temperature_range (uint32_t cct_min, uint32_t cct_max)
{
    XCAM_FAIL_RETURN (
        ERROR,
        (cct_min <= cct_max),
        false,
        "awb set wrong cct(%u, %u) parameters", cct_min, cct_max);

    AnalyzerHandler::HandlerLock lock(this);
    _params.cct_min = cct_min;
    _params.cct_max = cct_max;

    XCAM_LOG_DEBUG ("awb set cct range [%u, %u]", cct_min, cct_max);
    return true;
}

bool
AwbHandler::set_manual_gain (double gr, double r, double b, double gb)
{
    XCAM_FAIL_RETURN (
        ERROR,
        gr >= 0.0 && r >= 0.0 && b >= 0.0 && gb >= 0.0,
        false,
        "awb manual gain value must >= 0.0");

    AnalyzerHandler::HandlerLock lock(this);
    _params.gr_gain = gr;
    _params.r_gain = r;
    _params.b_gain = b;
    _params.gb_gain = gb;
    XCAM_LOG_DEBUG ("awb set manual gain value(gr:%.03f, r:%.03f, b:%.03f, gb:%.03f)", gr, r, b, gb);
    return true;
}

bool
AwbHandler::update_parameters (const XCamAwbParam &params)
{
    {
        AnalyzerHandler::HandlerLock lock (this);
        _params = params;
    }
    XCAM_LOG_DEBUG ("awb parameters updated");
    return true;
}

uint32_t
AwbHandler::get_current_estimate_cct ()
{
    AnalyzerHandler::HandlerLock lock(this);
    if (_params.mode == XCAM_AWB_MODE_MANUAL)
        return (_params.cct_max + _params.cct_min) / 2;
    return 0.0;
}

bool
AfHandler::update_parameters (const XCamAfParam &params)
{
    {
        AnalyzerHandler::HandlerLock lock (this);
        _params = params;
    }
    XCAM_LOG_DEBUG ("af parameters updated");
    return true;
}

CommonHandler::CommonHandler()
{
    reset_parameters ();
}

void
CommonHandler::reset_parameters ()
{
    xcam_mem_clear (_params);

    _params.is_manual_gamma = false;
    _params.nr_level = 0.0;
    _params.tnr_level = 0.0;
    _params.brightness = 0.0;
    _params.contrast = 0.0;
    _params.hue = 0.0;
    _params.saturation = 0.0;
    _params.sharpness = 0.0;
    _params.enable_dvs = false;
    _params.enable_gbce = false;
    _params.enable_night_mode = false;
}

bool CommonHandler::set_dvs (bool enable)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.enable_dvs = enable;

    XCAM_LOG_DEBUG ("common 3A enable dvs:%s", XCAM_BOOL2STR(enable));
    return true;
}

bool
CommonHandler::set_gbce (bool enable)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.enable_gbce = enable;

    XCAM_LOG_DEBUG ("common 3A enable gbce:%s", XCAM_BOOL2STR(enable));
    return true;
}

bool
CommonHandler::set_night_mode (bool enable)
{
    AnalyzerHandler::HandlerLock lock(this);
    _params.enable_night_mode = enable;

    XCAM_LOG_DEBUG ("common 3A enable night mode:%s", XCAM_BOOL2STR(enable));
    return true;
}

/* Picture quality */
bool
CommonHandler::set_noise_reduction_level (double level)
{
    XCAM_FAIL_RETURN (
        ERROR,
        level >= -1.0 && level < 1.0,
        false,
        "set NR levlel(%.03f) out of range[-1.0, 1.0]", level);

    AnalyzerHandler::HandlerLock lock(this);
    _params.nr_level = level;

    XCAM_LOG_DEBUG ("common 3A set NR level:%.03f", level);
    return true;
}

bool
CommonHandler::set_temporal_noise_reduction_level (double level)
{
    XCAM_FAIL_RETURN (
        ERROR,
        level >= -1.0 && level < 1.0,
        false,
        "set TNR levlel(%.03f) out of range[-1.0, 1.0]", level);

    AnalyzerHandler::HandlerLock lock(this);
    _params.tnr_level = level;

    XCAM_LOG_DEBUG ("common 3A set TNR level:%.03f", level);
    return true;
}

bool
CommonHandler::set_manual_brightness (double level)
{
    XCAM_FAIL_RETURN (
        ERROR,
        level >= -1.0 && level < 1.0,
        false,
        "set brightness levlel(%.03f) out of range[-1.0, 1.0]", level);

    AnalyzerHandler::HandlerLock lock(this);
    _params.brightness = level;

    XCAM_LOG_DEBUG ("common 3A set brightness level:%.03f", level);
    return true;
}

bool CommonHandler::set_manual_contrast (double level)
{
    XCAM_FAIL_RETURN (
        ERROR,
        level >= -1.0 && level < 1.0,
        false,
        "set contrast levlel(%.03f) out of range[-1.0, 1.0]", level);

    AnalyzerHandler::HandlerLock lock(this);
    _params.contrast = level;

    XCAM_LOG_DEBUG ("common 3A set contrast level:%.03f", level);
    return true;
}

bool CommonHandler::set_manual_hue (double level)
{
    XCAM_FAIL_RETURN (
        ERROR,
        level >= -1.0 && level < 1.0,
        false,
        "set hue levlel(%.03f) out of range[-1.0, 1.0]", level);

    AnalyzerHandler::HandlerLock lock(this);
    _params.hue = level;

    XCAM_LOG_DEBUG ("common 3A set hue level:%.03f", level);
    return true;
}

bool
CommonHandler::set_manual_saturation (double level)
{
    XCAM_FAIL_RETURN (
        ERROR,
        level >= -1.0 && level < 1.0,
        false,
        "set saturation levlel(%.03f) out of range[-1.0, 1.0]", level);

    AnalyzerHandler::HandlerLock lock(this);
    _params.saturation = level;

    XCAM_LOG_DEBUG ("common 3A set saturation level:%.03f", level);
    return true;
}

bool CommonHandler::set_manual_sharpness (double level)
{
    XCAM_FAIL_RETURN (
        ERROR,
        level >= -1.0 && level < 1.0,
        false,
        "set sharpness levlel(%.03f) out of range[-1.0, 1.0]", level);

    AnalyzerHandler::HandlerLock lock(this);
    _params.sharpness = level;

    XCAM_LOG_DEBUG ("common 3A set sharpness level:%.03f", level);
    return true;
}

bool
CommonHandler::set_gamma_table (double *r_table, double *g_table, double *b_table)
{
    AnalyzerHandler::HandlerLock lock(this);
    if (!r_table && ! g_table && !b_table) {
        _params.is_manual_gamma = false;
        XCAM_LOG_DEBUG ("common 3A disabled gamma");
        return true;
    }

    if (!r_table || !g_table || !b_table) {
        XCAM_LOG_ERROR ("common 3A gamma table parameters wrong");
        return false;
    }

    for (uint32_t i = 0; i < XCAM_GAMMA_TABLE_SIZE; ++i) {
        _params.r_gamma [i] = r_table [i];
        _params.g_gamma [i] = g_table [i];
        _params.b_gamma [i] = b_table [i];
    }
    _params.is_manual_gamma = true;

    XCAM_LOG_DEBUG ("common 3A enabled RGB gamma");
    return true;
}

bool
CommonHandler::set_color_effect (XCamColorEffect effect)
{
    // TODO validate the input

    AnalyzerHandler::HandlerLock lock(this);

    _params.color_effect = effect;

    XCAM_LOG_DEBUG ("common 3A set color effect");
    return true;
}

bool
CommonHandler::update_parameters (const XCamCommonParam &params)
{
    {
        AnalyzerHandler::HandlerLock lock (this);
        _params = params;
    }
    XCAM_LOG_DEBUG ("common parameters updated");
    return true;
}

};
