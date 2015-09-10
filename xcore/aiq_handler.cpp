/*
 * aiq_handler.cpp - AIQ handler
 *
 *  Copyright (c) 2012-2015 Intel Corporation
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
 * Author: Yan Zhang <yan.y.zhang@intel.com>
 */

#include "aiq_handler.h"
#include "x3a_isp_config.h"

#include <string.h>
#include <math.h>

#include "ia_isp_2_2.h"

#define MAX_STATISTICS_WIDTH 150
#define MAX_STATISTICS_HEIGHT 150

//#define USE_RGBS_GRID_WEIGHTING
#define USE_HIST_GRID_WEIGHTING

namespace XCam {

struct IspInputParameters {
    ia_aiq_frame_use            frame_use;
    ia_aiq_frame_params        *sensor_frame_params;
    ia_aiq_exposure_parameters *exposure_results;
    ia_aiq_awb_results         *awb_results;
    ia_aiq_gbce_results        *gbce_results;
    ia_aiq_pa_results          *pa_results;
#ifdef HAVE_AIQ_2_7
    ia_aiq_sa_results          *sa_results;
#endif
    int8_t                      manual_brightness;
    int8_t                      manual_contrast;
    int8_t                      manual_hue;
    int8_t                      manual_saturation;
    int8_t                      manual_sharpness;
    int8_t                      manual_nr_level;
    ia_isp_effect               effects;

    IspInputParameters ()
        : frame_use (ia_aiq_frame_use_preview)
        , sensor_frame_params (NULL)
        , exposure_results (NULL)
        , awb_results (NULL)
        , gbce_results (NULL)
        , pa_results (NULL)
#ifdef HAVE_AIQ_2_7
        , sa_results (NULL)
#endif
        , manual_brightness (0)
        , manual_contrast (0)
        , manual_hue (0)
        , manual_saturation (0)
        , manual_sharpness (0)
        , manual_nr_level (0)
        , effects (ia_isp_effect_none)
    {}
};

class IaIspAdaptor22
    : public IaIspAdaptor
{
public:
    IaIspAdaptor22 () {
        xcam_mem_clear (_input_params);
    }
    ~IaIspAdaptor22 () {
        if (_handle)
            ia_isp_2_2_deinit (_handle);
    }

    virtual bool init (
        const ia_binary_data *cpf,
        unsigned int max_width,
        unsigned int max_height,
        ia_cmc_t *cmc,
        ia_mkn *mkn);

    virtual bool convert_statistics (
        void *statistics,
        ia_aiq_rgbs_grid **out_rgbs_grid,
        ia_aiq_af_grid **out_af_grid);

    virtual bool run (
        const IspInputParameters *isp_input_params,
        ia_binary_data *output_data);

private:
    ia_isp_2_2_input_params  _input_params;

};

bool
IaIspAdaptor22::init (
    const ia_binary_data *cpf,
    unsigned int max_width,
    unsigned int max_height,
    ia_cmc_t *cmc,
    ia_mkn *mkn)
{
    xcam_mem_clear (_input_params);
    _input_params.isp_vamem_type = 1;
    _handle = ia_isp_2_2_init (cpf, max_width, max_height, cmc, mkn);
    XCAM_FAIL_RETURN (ERROR, _handle, false, "ia_isp 2.2 init failed");
    return true;
}

bool
IaIspAdaptor22::convert_statistics (
    void *statistics,
    ia_aiq_rgbs_grid **out_rgbs_grid,
    ia_aiq_af_grid **out_af_grid)
{
    ia_err err;
    err = ia_isp_2_2_statistics_convert (_handle, statistics, out_rgbs_grid, out_af_grid);
    XCAM_FAIL_RETURN (ERROR, err == ia_err_none, false, "ia_isp 2.2 convert stats failed");
    return true;
}

bool
IaIspAdaptor22::run (
    const IspInputParameters *isp_input_params,
    ia_binary_data *output_data)
{
    ia_err err;

    _input_params.frame_use = isp_input_params->frame_use;
    _input_params.sensor_frame_params = isp_input_params->sensor_frame_params;
    _input_params.exposure_results = isp_input_params->exposure_results;
    _input_params.awb_results = isp_input_params->awb_results;
    _input_params.gbce_results = isp_input_params->gbce_results;
    _input_params.pa_results = isp_input_params->pa_results;
#ifdef HAVE_AIQ_2_7
    _input_params.sa_results = isp_input_params->sa_results;
#endif
    _input_params.manual_brightness = isp_input_params->manual_brightness;
    _input_params.manual_contrast = isp_input_params->manual_contrast;
    _input_params.manual_hue = isp_input_params->manual_hue;
    _input_params.manual_saturation = isp_input_params->manual_saturation;
    _input_params.nr_setting.feature_level = ia_isp_feature_level_high;
    _input_params.nr_setting.strength = isp_input_params->manual_nr_level;
    _input_params.ee_setting.feature_level = ia_isp_feature_level_high;
    _input_params.ee_setting.strength = isp_input_params->manual_sharpness;
    _input_params.effects = isp_input_params->effects;

    err = ia_isp_2_2_run (_handle, &_input_params, output_data);
    XCAM_FAIL_RETURN (ERROR, err == ia_err_none, false, "ia_isp 2.2 run failed");
    return true;
}

#if 0

class IaIspAdaptor15
    : public IaIspAdaptor
{
public:
    IaIspAdaptor15 () {
        xcam_mem_clear (&_input_params);
    }
    ~IaIspAdaptor15 () {
        if (_handle)
            ia_isp_1_5_deinit (_handle);
    }
    virtual bool init (
        const ia_binary_data *cpf,
        unsigned int max_width,
        unsigned int max_height,
        ia_cmc_t *cmc,
        ia_mkn *mkn);
    virtual bool convert_statistics (
        void *statistics,
        ia_aiq_rgbs_grid **out_rgbs_grid,
        ia_aiq_af_grid **out_af_grid);
    virtual bool run (
        const IspInputParameters *isp_input_params,
        ia_binary_data *output_data);

private:
    ia_isp_1_5_input_params  _input_params;

};

bool
IaIspAdaptor15::init (
    const ia_binary_data *cpf,
    unsigned int max_width,
    unsigned int max_height,
    ia_cmc_t *cmc,
    ia_mkn *mkn)
{
    xcam_mem_clear (&_input_params);
    _input_params.isp_vamem_type = 1;
    _handle = ia_isp_1_5_init (cpf, max_width, max_height, cmc, mkn);
    XCAM_FAIL_RETURN (ERROR, _handle, false, "ia_isp 1.5 init failed");
    return true;
}

bool
IaIspAdaptor15::convert_statistics (
    void *statistics,
    ia_aiq_rgbs_grid **out_rgbs_grid,
    ia_aiq_af_grid **out_af_grid)
{
    ia_err err;
    err = ia_isp_1_5_statistics_convert (_handle, statistics, out_rgbs_grid, out_af_grid);
    XCAM_FAIL_RETURN (ERROR, err == ia_err_none, false, "ia_isp 1.5 convert stats failed");
    return true;
}

bool
IaIspAdaptor15::run (
    const IspInputParameters *isp_input_params,
    ia_binary_data *output_data)
{
    ia_err err;

    _input_params.frame_use = isp_input_params->frame_use;
    _input_params.sensor_frame_params = isp_input_params->sensor_frame_params;
    _input_params.exposure_results = isp_input_params->exposure_results;
    _input_params.awb_results = isp_input_params->awb_results;
    _input_params.gbce_results = isp_input_params->gbce_results;
    _input_params.pa_results = isp_input_params->pa_results;
    _input_params.manual_brightness = isp_input_params->manual_brightness;
    _input_params.manual_contrast = isp_input_params->manual_contrast;
    _input_params.manual_hue = isp_input_params->manual_hue;
    _input_params.manual_saturation = isp_input_params->manual_saturation;
    _input_params.nr_setting.feature_level = ia_isp_feature_level_high;
    _input_params.nr_setting.strength = isp_input_params->manual_nr_level;
    _input_params.ee_setting.feature_level = ia_isp_feature_level_high;
    _input_params.ee_setting.strength = isp_input_params->manual_sharpness;
    _input_params.effects = isp_input_params->effects;

    err = ia_isp_1_5_run (_handle, &_input_params, output_data);
    XCAM_FAIL_RETURN (ERROR, err == ia_err_none, false, "ia_isp 1.5 run failed");
    return true;
}

#endif

static double
_calculate_new_value_by_speed (double start, double end, double speed)
{
    XCAM_ASSERT (speed >= 0.0 && speed <= 1.0);
    static const double value_equal_range = 0.000001;

    if (fabs (end - start) <= value_equal_range)
        return end;
    return (start * (1.0 - speed) + end * speed);
}

static double
_imx185_sensor_gain_code_to_mutiplier (uint32_t code)
{
    /* 185 sensor code : DB = 160 : 48 */
    double db;
    db = code * 48.0 / 160.0;
    return pow (10.0, db / 20.0);
}

static uint32_t
_mutiplier_to_imx185_sensor_gain_code (double mutiplier)
{
    double db = log10 (mutiplier) * 20;
    if (db > 48)
        db = 48;
    return (uint32_t) (db * 160 / 48);
}

static uint32_t
_time_to_coarse_line (ia_aiq_exposure_sensor_descriptor *desc, uint32_t time_us)
{
    float value =  time_us * desc->pixel_clock_freq_mhz;

    value = (value + desc->pixel_periods_per_line / 2) / desc->pixel_periods_per_line;
    return (uint32_t)(value);
}

AiqAeHandler::AiqAeResult::AiqAeResult()
{
    xcam_mem_clear (ae_result);
    xcam_mem_clear (ae_exp_ret);
    xcam_mem_clear (aiq_exp_param);
    xcam_mem_clear (sensor_exp_param);
    xcam_mem_clear (weight_grid);
    xcam_mem_clear (flash_param);
}

void
AiqAeHandler::AiqAeResult::copy (ia_aiq_ae_results *result)
{
    XCAM_ASSERT (result);

    this->ae_result = *result;
    this->aiq_exp_param = *result->exposures[0].exposure;
    this->sensor_exp_param = *result->exposures[0].sensor_exposure;
    this->weight_grid = *result->weight_grid;
#ifdef HAVE_AIQ_2_7
    this->flash_param = result->flashes[0];
#else
    this->flash_param = *result->flash;
#endif

    this->ae_exp_ret.exposure = &this->aiq_exp_param;
    this->ae_exp_ret.sensor_exposure = &this->sensor_exp_param;
    this->ae_result.exposures = &this->ae_exp_ret;
    this->ae_result.weight_grid = &this->weight_grid;
#ifdef HAVE_AIQ_2_7
    this->ae_result.flashes[0] = this->flash_param;
#else
    this->ae_result.flash = &this->flash_param;
#endif
    this->ae_result.num_exposures = 1;
}

AiqAeHandler::AiqAeHandler (SmartPtr<AiqCompositor> &aiq_compositor)
    : _aiq_compositor (aiq_compositor)
    , _started (false)
{
    xcam_mem_clear (_ia_ae_window);
    xcam_mem_clear (_sensor_descriptor);
    xcam_mem_clear (_manual_limits);
    xcam_mem_clear (_input);
    _input.num_exposures = 1;
    _input.frame_use = _aiq_compositor->get_frame_use();
    _input.flash_mode = ia_aiq_flash_mode_off;
    _input.operation_mode = ia_aiq_ae_operation_mode_automatic;
    _input.metering_mode = ia_aiq_ae_metering_mode_evaluative;
    _input.priority_mode = ia_aiq_ae_priority_mode_normal;
    _input.flicker_reduction_mode = ia_aiq_ae_flicker_reduction_auto;
    _input.sensor_descriptor = NULL;
    _input.exposure_window = NULL;
    _input.exposure_coordinate = NULL;
    _input.ev_shift = 0.0;
    _input.manual_exposure_time_us = -1;
    _input.manual_analog_gain = -1.0;
    _input.manual_iso = -1.0;
    _input.aec_features = NULL;
    _input.manual_limits = &_manual_limits;
}

bool
AiqAeHandler::set_description (struct atomisp_sensor_mode_data *sensor_data)
{
    XCAM_ASSERT (sensor_data);

    _sensor_descriptor.pixel_clock_freq_mhz = sensor_data->vt_pix_clk_freq_mhz / 1000000.0f;
    _sensor_descriptor.pixel_periods_per_line = sensor_data->line_length_pck;
    _sensor_descriptor.line_periods_per_field = sensor_data->frame_length_lines;
    _sensor_descriptor.line_periods_vertical_blanking = sensor_data->frame_length_lines
            - (sensor_data->crop_vertical_end - sensor_data->crop_vertical_start + 1)
            / sensor_data->binning_factor_y;
    _sensor_descriptor.fine_integration_time_min = sensor_data->fine_integration_time_def;
    _sensor_descriptor.fine_integration_time_max_margin = sensor_data->line_length_pck - sensor_data->fine_integration_time_def;
    _sensor_descriptor.coarse_integration_time_min = sensor_data->coarse_integration_time_min;
    _sensor_descriptor.coarse_integration_time_max_margin = sensor_data->coarse_integration_time_max_margin;

    return true;
}

bool
AiqAeHandler::ensure_ia_parameters ()
{
    bool ret = true;
    ret = ret && ensure_ae_mode ();
    ret = ret && ensure_ae_metering_mode ();
    ret = ret && ensure_ae_priority_mode ();
    ret = ret && ensure_ae_flicker_mode ();
    ret = ret && ensure_ae_manual ();
    ret = ret && ensure_ae_ev_shift ();
    _input.sensor_descriptor = &_sensor_descriptor;
    return ret;
}

bool AiqAeHandler::ensure_ae_mode ()
{
    XCamAeMode mode = this->get_mode_unlock();
    switch (mode) {
    case XCAM_AE_MODE_AUTO:
    case XCAM_AE_MODE_MANUAL:
        _input.operation_mode = ia_aiq_ae_operation_mode_automatic;
        break;

    case XCAM_AE_MODE_NOT_SET:
    default:
        XCAM_LOG_ERROR("unsupported ae mode:%d", mode);
        return false;
    }
    return true;
}
bool AiqAeHandler::ensure_ae_metering_mode ()
{
    XCamAeMeteringMode mode = this->get_metering_mode_unlock();

    _input.exposure_window = NULL;

    switch (mode) {
    case XCAM_AE_METERING_MODE_AUTO:
        _input.metering_mode = ia_aiq_ae_metering_mode_evaluative;
        break;
    case XCAM_AE_METERING_MODE_SPOT:
    {
        _input.metering_mode = ia_aiq_ae_metering_mode_evaluative;
        const XCam3AWindow & window = this->get_window_unlock();
        if (window.x_end > window.x_start &&
                window.y_end > window.y_start) {
            _aiq_compositor->convert_window_to_ia(window, _ia_ae_window);
            _input.exposure_window = &_ia_ae_window;
        }
    }
    break;
    case XCAM_AE_METERING_MODE_CENTER:
        _input.metering_mode = ia_aiq_ae_metering_mode_center;
        break;
    case XCAM_AE_METERING_MODE_WEIGHTED_WINDOW:
    {
        _input.metering_mode = ia_aiq_ae_metering_mode_evaluative;
        const XCam3AWindow & weighted_window = this->get_window_unlock();

        XCAM_LOG_DEBUG ("ensure_ae_metering_mode weighted_window x_start = %d, y_start = %d, x_end = %d, y_end = %d ",
                        weighted_window.x_start, weighted_window.y_start, weighted_window.x_end, weighted_window.y_end);

        if (weighted_window.x_end > weighted_window.x_start &&
                weighted_window.y_end > weighted_window.y_start) {
            _aiq_compositor->convert_window_to_ia(weighted_window, _ia_ae_window);
            _input.exposure_window = &_ia_ae_window;
        }
    }
    break;
    default:
        XCAM_LOG_ERROR("unsupported ae mode:%d", mode);
        return false;
    }
    return true;
}

bool AiqAeHandler::ensure_ae_priority_mode ()
{
    _input.priority_mode = ia_aiq_ae_priority_mode_normal;
    return true;
}

bool AiqAeHandler::ensure_ae_flicker_mode ()
{
    XCamFlickerMode mode = this->get_flicker_mode_unlock ();
    switch (mode) {
    case XCAM_AE_FLICKER_MODE_AUTO:
        _input.flicker_reduction_mode = ia_aiq_ae_flicker_reduction_auto;
        break;
    case XCAM_AE_FLICKER_MODE_50HZ:
        _input.flicker_reduction_mode = ia_aiq_ae_flicker_reduction_50hz;
        break;
    case XCAM_AE_FLICKER_MODE_60HZ:
        _input.flicker_reduction_mode = ia_aiq_ae_flicker_reduction_60hz;
        break;
    case XCAM_AE_FLICKER_MODE_OFF:
        _input.flicker_reduction_mode = ia_aiq_ae_flicker_reduction_off;
        break;
    default:
        XCAM_LOG_ERROR ("flicker mode(%d) unknown", mode);
        return false;
    }
    return true;
}

bool AiqAeHandler::ensure_ae_manual ()
{
    if (this->get_mode_unlock () == XCAM_AE_MODE_MANUAL) {
        _input.manual_exposure_time_us = get_manual_exposure_time_unlock ();
        _input.manual_analog_gain = get_manual_analog_gain_unlock ();
    }
    else {
        _input.manual_exposure_time_us = -1;
        _input.manual_analog_gain = -1;
    }

    _input.manual_limits->manual_exposure_time_min =
        _sensor_descriptor.coarse_integration_time_min
        * _sensor_descriptor.pixel_periods_per_line
        / _sensor_descriptor.pixel_clock_freq_mhz;
    _input.manual_limits->manual_exposure_time_max =
        (_sensor_descriptor.line_periods_per_field - _sensor_descriptor.coarse_integration_time_max_margin)
        * _sensor_descriptor.pixel_periods_per_line
        / _sensor_descriptor.pixel_clock_freq_mhz;

    uint64_t exp_min_us = 0, exp_max_us = 0;
    get_exposure_time_range_unlock (exp_min_us, exp_max_us);
    if (exp_min_us && (int64_t)exp_min_us > _input.manual_limits->manual_exposure_time_min) {
        _input.manual_limits->manual_exposure_time_min = exp_min_us;
    }
    if (exp_max_us && (int64_t)exp_max_us < _input.manual_limits->manual_exposure_time_max) {
        _input.manual_limits->manual_exposure_time_max = exp_max_us;
    }

    _input.manual_limits->manual_frame_time_us_min = -1;
    _input.manual_limits->manual_frame_time_us_max = 1000000 / _aiq_compositor->get_framerate ();
    _input.manual_limits->manual_iso_min = -1;
    _input.manual_limits->manual_iso_max = -1;

    return true;
}

bool AiqAeHandler::ensure_ae_ev_shift ()
{
    _input.ev_shift = this->get_ev_shift_unlock();
    return true;
}

SmartPtr<X3aResult>
AiqAeHandler::pop_result ()
{
    //AnalyzerHandler::HandlerLock lock(this);

    X3aIspExposureResult *result = new X3aIspExposureResult(XCAM_IMAGE_PROCESS_ONCE);
    struct atomisp_exposure sensor;
    XCam3aResultExposure exposure;

    xcam_mem_clear (sensor);
    sensor.integration_time[0] = _result.sensor_exp_param.coarse_integration_time;
    sensor.integration_time[1] = _result.sensor_exp_param.fine_integration_time;
    sensor.gain[0] = _result.sensor_exp_param.analog_gain_code_global;
    sensor.gain[1] = _result.sensor_exp_param.digital_gain_global;
    result->set_isp_config (sensor);

    xcam_mem_clear (exposure);
    exposure.exposure_time = _result.aiq_exp_param.exposure_time_us;
    exposure.analog_gain = _result.aiq_exp_param.analog_gain;
    exposure.digital_gain = _result.aiq_exp_param.digital_gain;
    exposure.aperture = _result.aiq_exp_param.aperture_fn;
    result->set_standard_result (exposure);

    return result;
}

XCamReturn
AiqAeHandler::analyze (X3aResultList &output)
{
    ia_aiq  *ia_handle = NULL;
    ia_aiq_ae_results *ae_result = NULL;
    ia_aiq_exposure_sensor_parameters *cur_sensor_result = NULL;
    ia_err ia_error = ia_err_none;
    bool need_apply = false;
    SmartPtr<X3aResult> result;

    AnalyzerHandler::HandlerLock lock(this);

    if (!ensure_ia_parameters ()) {
        XCAM_LOG_ERROR ("AIQ AE ensure ia parameters failed");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ia_handle = _aiq_compositor->get_handle ();
    XCAM_ASSERT (ia_handle);
    ia_error = ia_aiq_ae_run (ia_handle, &_input, &ae_result);
    XCAM_FAIL_RETURN (ERROR, ia_error == ia_err_none, XCAM_RETURN_ERROR_AIQ, "AIQ run AE failed");

    cur_sensor_result = ae_result->exposures[0].sensor_exposure;

    if (!_started) {
        _result.copy (ae_result);
        _started = true;
        need_apply = true;
    } else {
        //TODO
        ia_aiq_exposure_sensor_parameters *last_sensor_res = &_result.sensor_exp_param;
        if (last_sensor_res->coarse_integration_time !=  cur_sensor_result->coarse_integration_time ||
                last_sensor_res->fine_integration_time !=  cur_sensor_result->fine_integration_time ||
                last_sensor_res->analog_gain_code_global !=  cur_sensor_result->analog_gain_code_global ||
                last_sensor_res->digital_gain_global !=  cur_sensor_result->digital_gain_global) {
            ia_aiq_exposure_sensor_parameters cur_cp_res = *cur_sensor_result;
            if (!manual_control_result (cur_cp_res, *last_sensor_res)) {
                XCAM_LOG_WARNING ("manual control AE result failed");
            }
            _result.copy (ae_result);
            _result.sensor_exp_param = cur_cp_res;
            need_apply = true;
        }
    }

    if (need_apply) {
        result = pop_result ();
        if (result.ptr())
            output.push_back (result);
    }

    return XCAM_RETURN_NO_ERROR;
}

bool
AiqAeHandler::manual_control_result (
    ia_aiq_exposure_sensor_parameters &cur_res,
    const ia_aiq_exposure_sensor_parameters &last_res)
{
    adjust_ae_speed (cur_res, last_res, this->get_speed_unlock());
    adjust_ae_limitation (cur_res);

    return true;
}

void
AiqAeHandler::adjust_ae_speed (
    ia_aiq_exposure_sensor_parameters &cur_res,
    const ia_aiq_exposure_sensor_parameters &last_res,
    double ae_speed)
{
    double last_gain, input_gain, ret_gain;
    ia_aiq_exposure_sensor_parameters tmp_res;

    if (XCAM_DOUBLE_EQUAL_AROUND(ae_speed, 1.0 ))
        return;
    xcam_mem_clear (tmp_res);
    tmp_res.coarse_integration_time = _calculate_new_value_by_speed (
                                          last_res.coarse_integration_time,
                                          cur_res.coarse_integration_time,
                                          ae_speed);

    last_gain = _imx185_sensor_gain_code_to_mutiplier (last_res.analog_gain_code_global);
    input_gain = _imx185_sensor_gain_code_to_mutiplier (cur_res.analog_gain_code_global);
    ret_gain = _calculate_new_value_by_speed (last_gain, input_gain, ae_speed);

    tmp_res.analog_gain_code_global = _mutiplier_to_imx185_sensor_gain_code (ret_gain);

    XCAM_LOG_DEBUG ("AE speed: from (shutter:%d, gain:%d[%.03f]) to (shutter:%d, gain:%d[%.03f])",
                    cur_res.coarse_integration_time, cur_res.analog_gain_code_global, input_gain,
                    tmp_res.coarse_integration_time, tmp_res.analog_gain_code_global, ret_gain);

    cur_res.coarse_integration_time = tmp_res.coarse_integration_time;
    cur_res.analog_gain_code_global = tmp_res.analog_gain_code_global;
}

void
AiqAeHandler::adjust_ae_limitation (ia_aiq_exposure_sensor_parameters &cur_res)
{
    ia_aiq_exposure_sensor_descriptor * desc = &_sensor_descriptor;
    uint64_t exposure_min = 0, exposure_max = 0;
    double analog_max = get_max_analog_gain_unlock ();
    uint32_t min_coarse_value = desc->coarse_integration_time_min;
    uint32_t max_coarse_value = desc->line_periods_per_field - desc->coarse_integration_time_max_margin;
    uint32_t value;

    get_exposure_time_range_unlock (exposure_min, exposure_max);

    if (exposure_min) {
        value = _time_to_coarse_line (desc, (uint32_t)exposure_min);
        min_coarse_value = (value > min_coarse_value) ? value : min_coarse_value;
    }
    if (cur_res.coarse_integration_time < min_coarse_value) {
        cur_res.coarse_integration_time = min_coarse_value;
    }

    if (exposure_max) {
        value = _time_to_coarse_line (desc, (uint32_t)exposure_max);
        max_coarse_value = (value < max_coarse_value) ? value : max_coarse_value;
    }
    if (cur_res.coarse_integration_time > max_coarse_value) {
        cur_res.coarse_integration_time = max_coarse_value;
    }

    if (analog_max >= 1.0) {
        /* limit gains */
        double gain = _imx185_sensor_gain_code_to_mutiplier (cur_res.analog_gain_code_global);
        if (gain > analog_max)
            cur_res.analog_gain_code_global = _mutiplier_to_imx185_sensor_gain_code (analog_max);
    }
}

XCamFlickerMode
AiqAeHandler::get_flicker_mode ()
{
    {
        AnalyzerHandler::HandlerLock lock(this);
    }
    return AeHandler::get_flicker_mode ();
}

int64_t
AiqAeHandler::get_current_exposure_time ()
{
    AnalyzerHandler::HandlerLock lock(this);

    return (int64_t)_result.aiq_exp_param.exposure_time_us;
}

double
AiqAeHandler::get_current_analog_gain ()
{
    AnalyzerHandler::HandlerLock lock(this);
    return (double)_result.aiq_exp_param.analog_gain;
}

double
AiqAeHandler::get_max_analog_gain ()
{
    {
        AnalyzerHandler::HandlerLock lock(this);
    }
    return AeHandler::get_max_analog_gain ();
}

XCamReturn
AiqAeHandler::set_RGBS_weight_grid (ia_aiq_rgbs_grid **out_rgbs_grid)
{
    AnalyzerHandler::HandlerLock lock(this);

    rgbs_grid_block *rgbs_grid_ptr = (*out_rgbs_grid)->blocks_ptr;
    uint32_t rgbs_grid_index = 0;
    uint16_t rgbs_grid_width = (*out_rgbs_grid)->grid_width;
    uint16_t rgbs_grid_height = (*out_rgbs_grid)->grid_height;

    XCAM_LOG_DEBUG ("rgbs_grid_width = %d, rgbs_grid_height = %d", rgbs_grid_width, rgbs_grid_height);

    uint64_t weight_sum = 0;

    uint32_t image_width = 0;
    uint32_t image_height = 0;
    _aiq_compositor->get_size (image_width, image_height);
    XCAM_LOG_DEBUG ("image_width = %d, image_height = %d", image_width, image_height);

    uint32_t hor_pixels_per_grid = (image_width + (rgbs_grid_width >> 1)) / rgbs_grid_width;
    uint32_t vert_pixels_per_gird = (image_height + (rgbs_grid_height >> 1)) / rgbs_grid_height;
    XCAM_LOG_DEBUG ("rgbs grid: %d x %d pixels per grid cell", hor_pixels_per_grid, vert_pixels_per_gird);

    XCam3AWindow weighted_window = this->get_window_unlock ();
    uint32_t weighted_grid_width = ((weighted_window.x_end - weighted_window.x_start + 1) +
                                    (hor_pixels_per_grid >> 1)) / hor_pixels_per_grid;
    uint32_t weighted_grid_height = ((weighted_window.y_end - weighted_window.y_start + 1) +
                                     (vert_pixels_per_gird >> 1)) / vert_pixels_per_gird;
    XCAM_LOG_DEBUG ("weighted_grid_width = %d, weighted_grid_height = %d", weighted_grid_width, weighted_grid_height);

    uint32_t *weighted_avg_gr = (uint32_t*)xcam_malloc0 (5 * weighted_grid_width * weighted_grid_height * sizeof(uint32_t));
    if (NULL == weighted_avg_gr) {
        return XCAM_RETURN_ERROR_MEM;
    }
    uint32_t *weighted_avg_r = weighted_avg_gr + (weighted_grid_width * weighted_grid_height);
    uint32_t *weighted_avg_b = weighted_avg_r + (weighted_grid_width * weighted_grid_height);
    uint32_t *weighted_avg_gb = weighted_avg_b + (weighted_grid_width * weighted_grid_height);
    uint32_t *weighted_sat = weighted_avg_gb + (weighted_grid_width * weighted_grid_height);

    for (uint32_t win_index = 0; win_index < XCAM_AE_MAX_METERING_WINDOW_COUNT; win_index++) {
        XCAM_LOG_DEBUG ("window start point(%d, %d), end point(%d, %d), weight = %d",
                        _params.window_list[win_index].x_start, _params.window_list[win_index].y_start,
                        _params.window_list[win_index].x_end, _params.window_list[win_index].y_end,
                        _params.window_list[win_index].weight);

        if ((_params.window_list[win_index].weight <= 0) ||
                (_params.window_list[win_index].x_start < 0) ||
                (_params.window_list[win_index].x_end > image_width) ||
                (_params.window_list[win_index].y_start < 0) ||
                (_params.window_list[win_index].y_end > image_height) ||
                (_params.window_list[win_index].x_start >= _params.window_list[win_index].x_end) ||
                (_params.window_list[win_index].y_start >= _params.window_list[win_index].y_end) ||
                (_params.window_list[win_index].x_end - _params.window_list[win_index].x_start > image_width) ||
                (_params.window_list[win_index].y_end - _params.window_list[win_index].y_start > image_height)) {
            XCAM_LOG_DEBUG ("skip window index = %d ", win_index);
            continue;
        }

        rgbs_grid_index = (_params.window_list[win_index].x_start +
                           (hor_pixels_per_grid >> 1)) / hor_pixels_per_grid +
                          ((_params.window_list[win_index].y_start + (vert_pixels_per_gird >> 1))
                           / vert_pixels_per_gird) * rgbs_grid_width;

        weight_sum += _params.window_list[win_index].weight;

        XCAM_LOG_DEBUG ("cumulate rgbs grid statistic, window index = %d ", win_index);
        for (uint32_t i = 0; i < weighted_grid_height; i++) {
            for (uint32_t j = 0; j < weighted_grid_width; j++) {
                weighted_avg_gr[j + i * weighted_grid_width] += rgbs_grid_ptr[rgbs_grid_index + j +
                        i * rgbs_grid_width].avg_gr * _params.window_list[win_index].weight;
                weighted_avg_r[j + i * weighted_grid_width] += rgbs_grid_ptr[rgbs_grid_index + j +
                        i * rgbs_grid_width].avg_r * _params.window_list[win_index].weight;
                weighted_avg_b[j + i * weighted_grid_width] += rgbs_grid_ptr[rgbs_grid_index + j +
                        i * rgbs_grid_width].avg_b * _params.window_list[win_index].weight;
                weighted_avg_gb[j + i * weighted_grid_width] += rgbs_grid_ptr[rgbs_grid_index + j +
                        i * rgbs_grid_width].avg_gb * _params.window_list[win_index].weight;
                weighted_sat[j + i * weighted_grid_width] += rgbs_grid_ptr[rgbs_grid_index + j +
                        i * rgbs_grid_width].sat * _params.window_list[win_index].weight;
            }
        }
    }
    XCAM_LOG_DEBUG ("sum of weighing factor = %d ", weight_sum);

    rgbs_grid_index = (weighted_window.x_start + (hor_pixels_per_grid >> 1)) / hor_pixels_per_grid +
                      (weighted_window.y_start + (vert_pixels_per_gird >> 1)) / vert_pixels_per_gird * rgbs_grid_width;
    for (uint32_t i = 0; i < weighted_grid_height; i++) {
        for (uint32_t j = 0; j < weighted_grid_width; j++) {
            rgbs_grid_ptr[rgbs_grid_index + j + i * rgbs_grid_width].avg_gr =
                weighted_avg_gr[j + i * weighted_grid_width] / weight_sum;
            rgbs_grid_ptr[rgbs_grid_index + j + i * rgbs_grid_width].avg_r =
                weighted_avg_r[j + i * weighted_grid_width] / weight_sum;
            rgbs_grid_ptr[rgbs_grid_index + j + i * rgbs_grid_width].avg_b =
                weighted_avg_b[j + i * weighted_grid_width] / weight_sum;
            rgbs_grid_ptr[rgbs_grid_index + j + i * rgbs_grid_width].avg_gb =
                weighted_avg_gb[j + i * weighted_grid_width] / weight_sum;
            rgbs_grid_ptr[rgbs_grid_index + j + i * rgbs_grid_width].sat =
                weighted_sat[j + i * weighted_grid_width] / weight_sum;
        }
    }

    xcam_free (weighted_avg_gr);

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
AiqAeHandler::set_hist_weight_grid (ia_aiq_hist_weight_grid **out_weight_grid)
{
    AnalyzerHandler::HandlerLock lock(this);

    uint16_t hist_grid_width = (*out_weight_grid)->width;
    uint16_t hist_grid_height = (*out_weight_grid)->height;
    uint32_t hist_grid_index = 0;

    unsigned char* weights_map_ptr = (*out_weight_grid)->weights;

    uint32_t image_width = 0;
    uint32_t image_height = 0;
    _aiq_compositor->get_size (image_width, image_height);

    uint32_t hor_pixels_per_grid = (image_width + (hist_grid_width >> 1)) / hist_grid_width;
    uint32_t vert_pixels_per_gird = (image_height + (hist_grid_height >> 1)) / hist_grid_height;
    XCAM_LOG_DEBUG ("hist weight grid: %d x %d pixels per grid cell", hor_pixels_per_grid, vert_pixels_per_gird);

    memset (weights_map_ptr, 0, hist_grid_width * hist_grid_height);

    for (uint32_t win_index = 0; win_index < XCAM_AE_MAX_METERING_WINDOW_COUNT; win_index++) {
        XCAM_LOG_DEBUG ("window start point(%d, %d), end point(%d, %d), weight = %d",
                        _params.window_list[win_index].x_start, _params.window_list[win_index].y_start,
                        _params.window_list[win_index].x_end, _params.window_list[win_index].y_end,
                        _params.window_list[win_index].weight);

        if ((_params.window_list[win_index].weight <= 0) ||
                (_params.window_list[win_index].weight > 15) ||
                (_params.window_list[win_index].x_start < 0) ||
                (_params.window_list[win_index].x_end > image_width) ||
                (_params.window_list[win_index].y_start < 0) ||
                (_params.window_list[win_index].y_end > image_height) ||
                (_params.window_list[win_index].x_start >= _params.window_list[win_index].x_end) ||
                (_params.window_list[win_index].y_start >= _params.window_list[win_index].y_end) ||
                (_params.window_list[win_index].x_end - _params.window_list[win_index].x_start > image_width) ||
                (_params.window_list[win_index].y_end - _params.window_list[win_index].y_start > image_height)) {
            XCAM_LOG_DEBUG ("skip window index = %d ", win_index);
            continue;
        }

        uint32_t weighted_grid_width =
            ((_params.window_list[win_index].x_end - _params.window_list[win_index].x_start + 1) +
             (hor_pixels_per_grid >> 1)) / hor_pixels_per_grid;
        uint32_t weighted_grid_height =
            ((_params.window_list[win_index].y_end - _params.window_list[win_index].y_start + 1) +
             (vert_pixels_per_gird >> 1)) / vert_pixels_per_gird;

        hist_grid_index = (_params.window_list[win_index].x_start + (hor_pixels_per_grid >> 1)) / hor_pixels_per_grid +
                          ((_params.window_list[win_index].y_start + (vert_pixels_per_gird >> 1)) /
                           vert_pixels_per_gird) * hist_grid_width;

        for (uint32_t i = 0; i < weighted_grid_height; i++) {
            for (uint32_t j = 0; j < weighted_grid_width; j++) {
                weights_map_ptr[hist_grid_index + j + i * hist_grid_width] = _params.window_list[win_index].weight;
            }
        }
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
AiqAeHandler::dump_hist_weight_grid (const ia_aiq_hist_weight_grid *weight_grid)
{
    XCAM_LOG_DEBUG ("E dump_hist_weight_grid");
    if (NULL == weight_grid) {
        return XCAM_RETURN_ERROR_PARAM;
    }

    uint16_t grid_width = weight_grid->width;
    uint16_t grid_height = weight_grid->height;

    for (uint32_t i = 0; i < grid_height; i++) {
        for (uint32_t j = 0; j < grid_width; j++) {
            printf ("%d  ", weight_grid->weights[j + i * grid_width]);
        }
        printf("\n");
    }

    XCAM_LOG_DEBUG ("X dump_hist_weight_grid");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
AiqAeHandler::dump_RGBS_grid (const ia_aiq_rgbs_grid *rgbs_grid)
{
    XCAM_LOG_DEBUG ("E dump_RGBS_grid");
    if (NULL == rgbs_grid) {
        return XCAM_RETURN_ERROR_PARAM;
    }

    uint16_t grid_width = rgbs_grid->grid_width;
    uint16_t grid_height = rgbs_grid->grid_height;

    printf("AVG B\n");
    for (uint32_t i = 0; i < grid_height; i++) {
        for (uint32_t j = 0; j < grid_width; j++) {
            printf ("%d  ", rgbs_grid->blocks_ptr[j + i * grid_width].avg_b);
        }
        printf("\n");
    }
    printf("AVG Gb\n");
    for (uint32_t i = 0; i < grid_height; i++) {
        for (uint32_t j = 0; j < grid_width; j++) {
            printf ("%d  ", rgbs_grid->blocks_ptr[j + i * grid_width].avg_gb);
        }
        printf("\n");
    }
    printf("AVG Gr\n");
    for (uint32_t i = 0; i < grid_height; i++) {
        for (uint32_t j = 0; j < grid_width; j++) {
            printf ("%d  ", rgbs_grid->blocks_ptr[j + i * grid_width].avg_gr);
        }
        printf("\n");
    }
    printf("AVG R\n");
    for (uint32_t i = 0; i < grid_height; i++) {
        for (uint32_t j = 0; j < grid_width; j++) {
            printf ("%d  ", rgbs_grid->blocks_ptr[j + i * grid_width].avg_r);
            //printf ("%d  ", rgbs_grid->blocks_ptr[j + i * grid_width].sat);
        }
        printf("\n");
    }

    XCAM_LOG_DEBUG ("X dump_RGBS_grid");
    return XCAM_RETURN_NO_ERROR;
}

AiqAwbHandler::AiqAwbHandler (SmartPtr<AiqCompositor> &aiq_compositor)
    : _aiq_compositor (aiq_compositor)
    , _started (false)
{
    xcam_mem_clear (_cct_range);
    xcam_mem_clear (_result);
    xcam_mem_clear (_history_result);
    xcam_mem_clear (_cct_range);
    xcam_mem_clear (_input);

    _input.frame_use = aiq_compositor->get_frame_use ();
    _input.scene_mode = ia_aiq_awb_operation_mode_auto;
    _input.manual_cct_range = NULL;
    _input.manual_white_coordinate = NULL;
}

XCamReturn
AiqAwbHandler::analyze (X3aResultList &output)
{
    ia_aiq  *ia_handle = NULL;
    ia_aiq_awb_results *awb_ret = NULL;
    ia_err ia_error = ia_err_none;

    XCAM_UNUSED (output);

    AnalyzerHandler::HandlerLock lock(this);

    if (!ensure_ia_parameters ()) {
        XCAM_LOG_ERROR ("AIQ AE ensure ia parameters failed");
        return XCAM_RETURN_ERROR_PARAM;
    }

    ia_handle = _aiq_compositor->get_handle ();
    XCAM_ASSERT (ia_handle);
    ia_error = ia_aiq_awb_run (ia_handle, &_input, &awb_ret);
    XCAM_FAIL_RETURN (ERROR, ia_error == ia_err_none, XCAM_RETURN_ERROR_AIQ, "AIQ run AWB failed");

    _result = *awb_ret;
    if (!_started) {
        _history_result = _result;
        _started = true;
    }
    adjust_speed (_history_result);
    _history_result = _result;

    return XCAM_RETURN_NO_ERROR;
}

bool
AiqAwbHandler::ensure_ia_parameters ()
{
    bool ret = true;

    _input.frame_use = _aiq_compositor->get_frame_use ();
    ret = ret && ensure_awb_mode ();
    return ret;
}

bool
AiqAwbHandler::ensure_awb_mode ()
{
    XCamAwbMode mode = get_mode_unlock();

    _input.manual_cct_range = NULL;
    _input.scene_mode = ia_aiq_awb_operation_mode_auto;

    switch (mode) {
    case XCAM_AWB_MODE_AUTO:
        _input.scene_mode = ia_aiq_awb_operation_mode_auto;
        break;
    case XCAM_AWB_MODE_MANUAL: {
        uint32_t cct_min = 0, cct_max = 0;
        get_cct_range_unlock (cct_min, cct_max);
        if (cct_min  && cct_max) {
            _input.scene_mode = ia_aiq_awb_operation_mode_manual_cct_range;
            _cct_range.max_cct = cct_min;
            _cct_range.min_cct = cct_max;
            _input.manual_cct_range = &_cct_range;
        } else
            _input.scene_mode = ia_aiq_awb_operation_mode_auto;
        break;
    }
    case XCAM_AWB_MODE_DAYLIGHT:
        _input.scene_mode = ia_aiq_awb_operation_mode_daylight;
        break;
    case XCAM_AWB_MODE_SUNSET:
        _input.scene_mode = ia_aiq_awb_operation_mode_sunset;
        break;
    case XCAM_AWB_MODE_CLOUDY:
        _input.scene_mode = ia_aiq_awb_operation_mode_partly_overcast;
        break;
    case XCAM_AWB_MODE_TUNGSTEN:
        _input.scene_mode = ia_aiq_awb_operation_mode_incandescent;
        break;
    case XCAM_AWB_MODE_FLUORESCENT:
        _input.scene_mode = ia_aiq_awb_operation_mode_fluorescent;
        break;
    case XCAM_AWB_MODE_WARM_FLUORESCENT:
        _input.scene_mode = ia_aiq_awb_operation_mode_incandescent;
        break;
    case XCAM_AWB_MODE_SHADOW:
        _input.scene_mode = ia_aiq_awb_operation_mode_fully_overcast;
        break;
    case XCAM_AWB_MODE_WARM_INCANDESCENT:
        _input.scene_mode = ia_aiq_awb_operation_mode_incandescent;
        break;
    case XCAM_AWB_MODE_NOT_SET:
        break;

    default:
        XCAM_LOG_ERROR ("unknown or unsupported AWB mode(%d)", mode);
        return false;
    }
    return true;
}

void
AiqAwbHandler::adjust_speed (const ia_aiq_awb_results &last_ret)
{
    _result.final_r_per_g =
        _calculate_new_value_by_speed (
            last_ret.final_r_per_g, _result.final_r_per_g, get_speed_unlock ());
    _result.final_b_per_g =
        _calculate_new_value_by_speed (
            last_ret.final_b_per_g, _result.final_b_per_g, get_speed_unlock ());
}

uint32_t
AiqAwbHandler::get_current_estimate_cct ()
{
    AnalyzerHandler::HandlerLock lock(this);
    return (uint32_t)_result.cct_estimate;
}

XCamReturn
AiqAfHandler::analyze (X3aResultList &output)
{
    // TODO
    XCAM_UNUSED (output);
    return XCAM_RETURN_NO_ERROR;
}

AiqCommonHandler::AiqCommonHandler (SmartPtr<AiqCompositor> &aiq_compositor)
    : _aiq_compositor (aiq_compositor)
    , _gbce_result (NULL)
{
}


XCamReturn
AiqCommonHandler::analyze (X3aResultList &output)
{
    ia_aiq  *ia_handle = NULL;
    ia_aiq_gbce_results *gbce_result = NULL;
    ia_err ia_error = ia_err_none;

    XCAM_UNUSED (output);

    AnalyzerHandler::HandlerLock lock(this);

    ia_aiq_gbce_input_params gbce_input;
    xcam_mem_clear (gbce_input);
    if (has_gbce_unlock()) {
        gbce_input.gbce_level = ia_aiq_gbce_level_use_tuning;
    }
    else {
        gbce_input.gbce_level = ia_aiq_gbce_level_bypass;
    }
    gbce_input.frame_use = _aiq_compositor->get_frame_use ();
    gbce_input.ev_shift = _aiq_compositor->get_ae_ev_shift_unlock ();
    ia_handle = _aiq_compositor->get_handle ();
    XCAM_ASSERT (ia_handle);
    ia_error = ia_aiq_gbce_run (ia_handle, &gbce_input, &gbce_result);

    XCAM_FAIL_RETURN (ERROR, ia_error == ia_err_none, XCAM_RETURN_ERROR_AIQ, "AIQ run GBCE failed");

    //TODO, need copy GBCE result out, not just assign
    _gbce_result = gbce_result;

    return XCAM_RETURN_NO_ERROR;
}

class CmcParser {
public:
    explicit CmcParser (ia_binary_data &cpf)
    {
        _cmc = ia_cmc_parser_init (&cpf);
    }
    ~CmcParser ()
    {
        if (_cmc)
            ia_cmc_parser_deinit (_cmc);
    }
    ia_cmc_t *get() {
        return _cmc;
    }

private:
    ia_cmc_t *_cmc;
};

void
AiqCompositor::convert_window_to_ia (const XCam3AWindow &window, ia_rectangle &ia_window)
{
    ia_rectangle source;
    ia_coordinate_system source_system;
    ia_coordinate_system target_system = {IA_COORDINATE_TOP, IA_COORDINATE_LEFT, IA_COORDINATE_BOTTOM, IA_COORDINATE_RIGHT};

    source_system.left = 0;
    source_system.top = 0;
    source_system.right = this->_width;
    source_system.bottom = this->_height;
    XCAM_ASSERT (_width && _height);

    source.left = window.x_start;
    source.top = window.y_start;
    source.right = window.x_end;
    source.bottom = window.y_end;
    ia_coordinate_convert_rect (&source_system, &source, &target_system, &ia_window);
}

AiqCompositor::AiqCompositor ()
    : _ia_handle (NULL)
    , _ia_mkn (NULL)
    , _pa_result (NULL)
#ifdef HAVE_AIQ_2_7
    , _sa_result (NULL)
#endif
    , _frame_use (ia_aiq_frame_use_video)
    , _width (0)
    , _height (0)
{
    xcam_mem_clear (_frame_params);
}

AiqCompositor::~AiqCompositor ()
{
}

bool
AiqCompositor::open (ia_binary_data &cpf)
{
    CmcParser cmc (cpf);

    _ia_mkn = ia_mkn_init (ia_mkn_cfg_compression, 32000, 100000);
    _ia_handle =
        ia_aiq_init (
            &cpf, NULL, NULL,
            MAX_STATISTICS_WIDTH, MAX_STATISTICS_HEIGHT,
            1, //max_num_stats_in
            cmc.get(),
            _ia_mkn);

    if (_ia_handle == NULL) {
        XCAM_LOG_WARNING ("AIQ init failed");
        return false;
    }

    _adaptor = new IaIspAdaptor22;
    XCAM_ASSERT (_adaptor.ptr());
    if (!_adaptor->init (&cpf, MAX_STATISTICS_WIDTH, MAX_STATISTICS_HEIGHT, cmc.get(), _ia_mkn)) {
        XCAM_LOG_WARNING ("AIQ isp adaptor init failed");
        return false;
    }

    _pa_result = NULL;
#ifdef HAVE_AIQ_2_7
    _sa_result = NULL;
#endif

    XCAM_LOG_DEBUG ("Aiq compositor opened");
    return true;
}

void
AiqCompositor::close ()
{
    _adaptor.release ();
    if (_ia_handle) {
        ia_aiq_deinit (_ia_handle);
        _ia_handle = NULL;
    }

    if (_ia_mkn) {
        ia_mkn_uninit (_ia_mkn);
        _ia_mkn = NULL;
    }

    _ae_handler.release ();
    _awb_handler.release ();
    _af_handler.release ();
    _common_handler.release ();

    _pa_result = NULL;
#ifdef HAVE_AIQ_2_7
    _sa_result = NULL;
#endif

    XCAM_LOG_DEBUG ("Aiq compositor closed");
}

bool
AiqCompositor::set_sensor_mode_data (struct atomisp_sensor_mode_data *sensor_mode)
{
    _frame_params.horizontal_crop_offset = sensor_mode->crop_horizontal_start;
    _frame_params.vertical_crop_offset = sensor_mode->crop_vertical_start;
    _frame_params.cropped_image_height = sensor_mode->crop_vertical_end - sensor_mode->crop_vertical_start + 1;
    _frame_params.cropped_image_width = sensor_mode->crop_horizontal_end - sensor_mode->crop_horizontal_start + 1;

    /* hard code to 254? */
    _frame_params.horizontal_scaling_denominator = 254;
    _frame_params.vertical_scaling_denominator = 254;

    if ((_frame_params.cropped_image_width == 0) || (_frame_params.cropped_image_height == 0)) {
        _frame_params.horizontal_scaling_numerator = 0;
        _frame_params.vertical_scaling_numerator = 0;
    } else {
        _frame_params.horizontal_scaling_numerator =
            sensor_mode->output_width * 254 * sensor_mode->binning_factor_x / _frame_params.cropped_image_width;
        _frame_params.vertical_scaling_numerator =
            sensor_mode->output_height * 254 * sensor_mode->binning_factor_y / _frame_params.cropped_image_height;
    }

    if (!_ae_handler->set_description (sensor_mode)) {
        XCAM_LOG_WARNING ("AIQ set ae description failed");
        return XCAM_RETURN_ERROR_AIQ;
    }
    return true;
}

bool
AiqCompositor::set_3a_stats (SmartPtr<X3aIspStatistics> &stats)
{
    ia_aiq_statistics_input_params aiq_stats_input;
    ia_aiq_rgbs_grid *rgbs_grids = NULL;
    ia_aiq_af_grid *af_grids = NULL;

    xcam_mem_clear (aiq_stats_input);
    aiq_stats_input.frame_timestamp = stats->get_timestamp();
    aiq_stats_input.frame_id = stats->get_timestamp() + 1;
    aiq_stats_input.rgbs_grids = (const ia_aiq_rgbs_grid **)&rgbs_grids;
    aiq_stats_input.num_rgbs_grids = 1;
    aiq_stats_input.af_grids = (const ia_aiq_af_grid **)(&af_grids);
    aiq_stats_input.num_af_grids = 1;

    aiq_stats_input.frame_af_parameters = NULL;
    aiq_stats_input.external_histograms = NULL;
    aiq_stats_input.num_external_histograms = 0;
    aiq_stats_input.camera_orientation = ia_aiq_camera_orientation_unknown;

    if (_pa_result)
        aiq_stats_input.frame_pa_parameters = _pa_result;

#ifdef HAVE_AIQ_2_7
    if (_sa_result)
        aiq_stats_input.frame_sa_parameters = _sa_result;
#endif

    if (_ae_handler->is_started()) {
#ifdef USE_HIST_GRID_WEIGHTING
        if (XCAM_AE_METERING_MODE_WEIGHTED_WINDOW == _ae_handler->get_metering_mode ()) {
            ia_aiq_ae_results* ae_result = _ae_handler->get_result ();

            if (XCAM_RETURN_NO_ERROR != _ae_handler->set_hist_weight_grid (&(ae_result->weight_grid))) {
                XCAM_LOG_ERROR ("ae handler set hist weight grid failed");
            }
        }
#endif
        aiq_stats_input.frame_ae_parameters = _ae_handler->get_result ();
        //_ae_handler->dump_hist_weight_grid (aiq_stats_input.frame_ae_parameters->weight_grid);
    }
    //if (_awb_handler->is_started())
    //    aiq_stats_input.frame_awb_parameters = _awb_handler->get_result();

    if (!_adaptor->convert_statistics (stats->get_isp_stats(), &rgbs_grids, &af_grids)) {
        XCAM_LOG_WARNING ("ia isp adaptor convert 3a stats failed");
        return false;
    }

    if (XCAM_AE_METERING_MODE_WEIGHTED_WINDOW == _ae_handler->get_metering_mode ()) {
#ifdef USE_RGBS_GRID_WEIGHTING
        if (XCAM_RETURN_NO_ERROR != _ae_handler->set_RGBS_weight_grid(&rgbs_grids)) {
            XCAM_LOG_ERROR ("ae handler update RGBS weighted statistic failed");
        }
        //_ae_handler->dump_RGBS_grid (*(aiq_stats_input.rgbs_grids));
#endif
    }
    XCAM_LOG_DEBUG ("statistics grid info, width:%u, height:%u, blk_r:%u, blk_b:%u, blk_gr:%u, blk_gb:%u",
                    aiq_stats_input.rgbs_grids[0]->grid_width,
                    aiq_stats_input.rgbs_grids[0]->grid_height,
                    aiq_stats_input.rgbs_grids[0]->blocks_ptr->avg_r,
                    aiq_stats_input.rgbs_grids[0]->blocks_ptr->avg_b,
                    aiq_stats_input.rgbs_grids[0]->blocks_ptr->avg_gr,
                    aiq_stats_input.rgbs_grids[0]->blocks_ptr->avg_gb);

    if (ia_aiq_statistics_set(get_handle (), &aiq_stats_input) != ia_err_none) {
        XCAM_LOG_ERROR ("Aiq set statistic failed");
        return false;
    }
    return true;
}

XCamReturn AiqCompositor::convert_color_effect (IspInputParameters &isp_input)
{
    AiqCommonHandler *aiq_common = _common_handler.ptr();

    switch (aiq_common->get_color_effect()) {
    case XCAM_COLOR_EFFECT_NONE:
        isp_input.effects = ia_isp_effect_none;
        break;
    case XCAM_COLOR_EFFECT_SKY_BLUE:
        isp_input.effects = ia_isp_effect_sky_blue;
        break;
    case XCAM_COLOR_EFFECT_SKIN_WHITEN_LOW:
        isp_input.effects = ia_isp_effect_skin_whiten_low;
        break;
    case XCAM_COLOR_EFFECT_SKIN_WHITEN:
        isp_input.effects = ia_isp_effect_skin_whiten;
        break;
    case XCAM_COLOR_EFFECT_SKIN_WHITEN_HIGH:
        isp_input.effects = ia_isp_effect_skin_whiten_high;
        break;
    case XCAM_COLOR_EFFECT_SEPIA:
        isp_input.effects = ia_isp_effect_sepia;
        break;
    case XCAM_COLOR_EFFECT_NEGATIVE:
        isp_input.effects = ia_isp_effect_negative;
        break;
    case XCAM_COLOR_EFFECT_GRAYSCALE:
        isp_input.effects = ia_isp_effect_grayscale;
        break;
    default:
        isp_input.effects = ia_isp_effect_none;
        break;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
AiqCompositor::apply_gamma_table (struct atomisp_parameters *isp_param)
{
    if (_common_handler->_params.is_manual_gamma) {
        int i;

        if (isp_param->r_gamma_table) {
            isp_param->r_gamma_table->vamem_type = 1; //IA_CSS_VAMEM_TYPE_2 = 1;
            for (i = 0; i < XCAM_GAMMA_TABLE_SIZE; ++i) {
                // change from double to u0.12
                isp_param->r_gamma_table->data.vamem_2[i] =
                    (uint32_t) (_common_handler->_params.r_gamma[i] * 4096.0);
            }
            isp_param->r_gamma_table->data.vamem_2[256] = 4091;
        }

        if (isp_param->g_gamma_table) {
            isp_param->g_gamma_table->vamem_type = 1; //IA_CSS_VAMEM_TYPE_2 = 1;
            for (i = 0; i < XCAM_GAMMA_TABLE_SIZE; ++i) {
                // change from double to u0.12
                isp_param->g_gamma_table->data.vamem_2[i] =
                    (uint32_t) (_common_handler->_params.g_gamma[i] * 4096.0);
            }
            isp_param->g_gamma_table->data.vamem_2[256] = 4091;
        }

        if (isp_param->b_gamma_table) {
            isp_param->b_gamma_table->vamem_type = 1; //IA_CSS_VAMEM_TYPE_2 = 1;
            for (i = 0; i < XCAM_GAMMA_TABLE_SIZE; ++i) {
                // change from double to u0.12
                isp_param->b_gamma_table->data.vamem_2[i] =
                    (uint32_t) (_common_handler->_params.b_gamma[i] * 4096.0);
            }
            isp_param->b_gamma_table->data.vamem_2[256] = 4091;
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
AiqCompositor::apply_night_mode (struct atomisp_parameters *isp_param)
{
    static const struct atomisp_cc_config night_yuv2rgb_cc_config = {
        10,
        {   1 << 10, 0, 0,  /* 1.0, 0, 0 */
            1 << 10, 0, 0,  /* 1.0, 0, 0 */
            1 << 10, 0, 0
        }
    }; /* 1.0, 0, 0 */
    static const struct atomisp_wb_config night_wb_config = {
        1,
        1 << 15, 1 << 15, 1 << 15, 1 << 15
    }; /* 1.0, 1.0, 1.0, 1.0*/

    if (isp_param->ctc_config)
        isp_param->ctc_config = NULL;

    *isp_param->wb_config = night_wb_config;
    *isp_param->yuv2rgb_cc_config = night_yuv2rgb_cc_config;

    return XCAM_RETURN_NO_ERROR;
}

double
AiqCompositor::calculate_value_by_factor (double factor, double min, double mid, double max)
{
    XCAM_ASSERT (factor >= -1.0 && factor <= 1.0);
    XCAM_ASSERT (min <= mid && max >= mid);

    if (factor >= 0.0)
        return (mid * (1.0 - factor) + max * factor);
    else
        return (mid * (1.0 + factor) + min * (-factor));
}

XCamReturn
AiqCompositor::limit_nr_levels (struct atomisp_parameters *isp_param)
{
#define NR_MIN_FACOTR 0.1
#define NR_MAX_FACOTR 6.0
#define NR_MID_FACOTR 1.0
    SmartPtr<AiqCommonHandler> aiq_common = _common_handler;

    if (!XCAM_DOUBLE_EQUAL_AROUND (aiq_common->_params.nr_level, 0.0)) {
        double nr_factor;
        nr_factor = calculate_value_by_factor (
                        aiq_common->_params.nr_level, NR_MIN_FACOTR, NR_MID_FACOTR, NR_MAX_FACOTR);
        if (isp_param->nr_config) {
            isp_param->nr_config->bnr_gain =
                XCAM_MIN (isp_param->nr_config->bnr_gain * nr_factor, 65535);
            isp_param->nr_config->ynr_gain =
                XCAM_MIN (isp_param->nr_config->ynr_gain * nr_factor, 65535);
        }
        if (isp_param->cnr_config) {
            isp_param->cnr_config->sense_gain_vy =
                XCAM_MIN (isp_param->cnr_config->sense_gain_vy * nr_factor, 8191);
            isp_param->cnr_config->sense_gain_vu =
                XCAM_MIN (isp_param->cnr_config->sense_gain_vu * nr_factor, 8191);
            isp_param->cnr_config->sense_gain_vv =
                XCAM_MIN (isp_param->cnr_config->sense_gain_vv * nr_factor, 8191);
            isp_param->cnr_config->sense_gain_hy =
                XCAM_MIN (isp_param->cnr_config->sense_gain_hy * nr_factor, 8191);
            isp_param->cnr_config->sense_gain_hu =
                XCAM_MIN (isp_param->cnr_config->sense_gain_hu * nr_factor, 8191);
            isp_param->cnr_config->sense_gain_hv =
                XCAM_MIN (isp_param->cnr_config->sense_gain_hv * nr_factor, 8191);
        }
    }

    if (!XCAM_DOUBLE_EQUAL_AROUND (aiq_common->_params.tnr_level, 0.0)) {
        double tnr_factor;
        tnr_factor = calculate_value_by_factor (
                         aiq_common->_params.tnr_level, NR_MIN_FACOTR, NR_MID_FACOTR, NR_MAX_FACOTR);
        if (isp_param->tnr_config) {
            isp_param->tnr_config->gain =
                XCAM_MIN (isp_param->tnr_config->gain * tnr_factor, 65535);
            if (XCAM_DOUBLE_EQUAL_AROUND (aiq_common->_params.tnr_level, 1.0)) {
                isp_param->tnr_config->gain = 65535;
                isp_param->tnr_config->threshold_y = 0;
                isp_param->tnr_config->threshold_uv = 0;
            }
        }
        XCAM_LOG_DEBUG ("set TNR gain:%u", isp_param->tnr_config->gain);
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn AiqCompositor::integrate (X3aResultList &results)
{
    IspInputParameters isp_params;
    ia_aiq_pa_input_params pa_input;
    ia_aiq_pa_results *pa_result = NULL;
#ifdef HAVE_AIQ_2_7
    ia_aiq_sa_input_params sa_input;
    ia_aiq_sa_results *sa_result = NULL;
#endif
    ia_err ia_error = ia_err_none;
    ia_binary_data output;
    AiqAeHandler *aiq_ae = _ae_handler.ptr();
    AiqAwbHandler *aiq_awb = _awb_handler.ptr();
    AiqAfHandler *aiq_af = _af_handler.ptr();
    AiqCommonHandler *aiq_common = _common_handler.ptr();
    struct atomisp_parameters *isp_3a_result = NULL;
    SmartPtr<X3aResult> isp_results;

    XCAM_FAIL_RETURN (
        ERROR,
        aiq_ae && aiq_awb && aiq_af && aiq_common,
        XCAM_RETURN_ERROR_PARAM,
        "handlers are not AIQ inherited");

    xcam_mem_clear (pa_input);
#ifndef HAVE_AIQ_2_7
    pa_input.frame_use = _frame_use;
    pa_input.sensor_frame_params = &_frame_params;
#endif
    if (aiq_ae->is_started())
        pa_input.exposure_params = (aiq_ae->get_result ())->exposures[0].exposure;
    pa_input.color_gains = NULL;

    if (aiq_common->_params.enable_night_mode) {
        pa_input.awb_results = NULL;
        isp_params.effects = ia_isp_effect_grayscale;
    }
    else {
        pa_input.awb_results = aiq_awb->get_result ();
    }

    ia_error = ia_aiq_pa_run (_ia_handle, &pa_input, &pa_result);
    if (ia_error != ia_err_none) {
        XCAM_LOG_WARNING ("AIQ pa run failed"); // but not return error
    }
    _pa_result = pa_result;

    if (aiq_awb->get_mode_unlock () == XCAM_AWB_MODE_MANUAL) {
        if (XCAM_DOUBLE_EQUAL_AROUND (aiq_awb->_params.gr_gain, 0.0) ||
                XCAM_DOUBLE_EQUAL_AROUND (aiq_awb->_params.r_gain, 0.0)  ||
                XCAM_DOUBLE_EQUAL_AROUND (aiq_awb->_params.b_gain, 0.0)  ||
                XCAM_DOUBLE_EQUAL_AROUND (aiq_awb->_params.gb_gain, 0.0)) {
            XCAM_LOG_DEBUG ("Zero gain would cause ISP division fatal error");
        }
        else {
#ifdef HAVE_AIQ_2_7
            _pa_result->color_gains.gr = aiq_awb->_params.gr_gain;
            _pa_result->color_gains.r = aiq_awb->_params.r_gain;
            _pa_result->color_gains.b = aiq_awb->_params.b_gain;
            _pa_result->color_gains.gb = aiq_awb->_params.gb_gain;
#else
            _pa_result->color_gains[0] = aiq_awb->_params.gr_gain;
            _pa_result->color_gains[1] = aiq_awb->_params.r_gain;
            _pa_result->color_gains[2] = aiq_awb->_params.b_gain;
            _pa_result->color_gains[3] = aiq_awb->_params.gb_gain;
#endif
        }
    }

#ifdef HAVE_AIQ_2_7
    xcam_mem_clear (sa_input);
    sa_input.frame_use = _frame_use;
    sa_input.sensor_frame_params = &_frame_params;
    if (aiq_common->_params.enable_night_mode) {
        sa_input.awb_results = NULL;
    }
    else {
        sa_input.awb_results = aiq_awb->get_result ();
    }

    ia_error = ia_aiq_sa_run (_ia_handle, &sa_input, &sa_result);
    if (ia_error != ia_err_none) {
        XCAM_LOG_WARNING ("AIQ sa run failed"); // but not return error
    }
    _sa_result = sa_result;
#endif

    isp_params.frame_use = _frame_use;
    isp_params.awb_results = aiq_awb->get_result ();
    if (aiq_ae->is_started())
        isp_params.exposure_results = (aiq_ae->get_result ())->exposures[0].exposure;
    isp_params.gbce_results = aiq_common->get_gbce_result ();
    isp_params.sensor_frame_params = &_frame_params;
    isp_params.pa_results = pa_result;
#ifdef HAVE_AIQ_2_7
    isp_params.sa_results = sa_result;
#endif

    isp_params.manual_brightness = (int8_t)(aiq_common->get_brightness_unlock() * 128.0);
    isp_params.manual_contrast = (int8_t)(aiq_common->get_contrast_unlock() * 128.0);
    isp_params.manual_saturation = (int8_t)(aiq_common->get_saturation_unlock() * 128.0);
    isp_params.manual_hue = (int8_t)(aiq_common->get_hue_unlock() * 128.0);
    isp_params.manual_sharpness = (int8_t)(aiq_common->get_sharpness_unlock() * 128.0);
    isp_params.manual_nr_level = (int8_t)(aiq_common->get_nr_level_unlock() * 128.0);

    convert_color_effect(isp_params);

    xcam_mem_clear (output);
    if (!_adaptor->run (&isp_params, &output)) {
        XCAM_LOG_ERROR("Aiq to isp adaptor running failed");
        return XCAM_RETURN_ERROR_ISP;
    }
    isp_3a_result = ((struct atomisp_parameters *)output.data);
    apply_gamma_table (isp_3a_result);
    limit_nr_levels (isp_3a_result);
    if (aiq_common->_params.enable_night_mode)
    {
        apply_night_mode (isp_3a_result);
    }

    isp_results = generate_3a_configs (isp_3a_result);
    results.push_back (isp_results);
    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<X3aResult>
AiqCompositor::generate_3a_configs (struct atomisp_parameters *parameters)
{
    SmartPtr<X3aResult> ret;

    X3aAtomIspParametersResult *x3a_result =
        new X3aAtomIspParametersResult (XCAM_IMAGE_PROCESS_ONCE);
    x3a_result->set_isp_config (*parameters);
    ret = x3a_result;
    return ret;
}

void
AiqCompositor::set_ae_handler (SmartPtr<AiqAeHandler> &handler)
{
    XCAM_ASSERT (!_ae_handler.ptr());
    _ae_handler = handler;
}

void
AiqCompositor::set_awb_handler (SmartPtr<AiqAwbHandler> &handler)
{
    XCAM_ASSERT (!_awb_handler.ptr());
    _awb_handler = handler;
}

void
AiqCompositor::set_af_handler (SmartPtr<AiqAfHandler> &handler)
{
    XCAM_ASSERT (!_af_handler.ptr());
    _af_handler = handler;
}

void
AiqCompositor::set_common_handler (SmartPtr<AiqCommonHandler> &handler)
{
    XCAM_ASSERT (!_common_handler.ptr());
    _common_handler = handler;
}


};
