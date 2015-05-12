/*
 * aiq_handler.h - AIQ handler
 *
 *  Copyright (c) 2014-2015  Intel Corporation
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

#ifndef XCAM_AIQ_HANDLER_H
#define XCAM_AIQ_HANDLER_H

#include "xcam_utils.h"
#include "handler_interface.h"
#include "x3a_statistics_queue.h"
#include "ia_types.h"
#include "ia_aiq_types.h"
#include "ia_cmc_parser.h"
#include "ia_mkn_encoder.h"
#include "ia_aiq.h"
#include "ia_coordinate.h"

typedef struct ia_isp_t ia_isp;

namespace XCam {

class AiqCompositor;
struct IspInputParameters;

class IaIspAdaptor {
public:
    explicit IaIspAdaptor()
        : _handle (NULL)
    {}
    virtual ~IaIspAdaptor() {}

    virtual bool init (
        const ia_binary_data *cpf,
        unsigned int max_width,
        unsigned int max_height,
        ia_cmc_t *cmc,
        ia_mkn *mkn) = 0;
    virtual bool convert_statistics (
        void *statistics,
        ia_aiq_rgbs_grid **out_rgbs_grid,
        ia_aiq_af_grid **out_af_grid) = 0;
    virtual bool run (
        const IspInputParameters *isp_input_params,
        ia_binary_data *output_data) = 0;

private:
    XCAM_DEAD_COPY (IaIspAdaptor);

protected:
    ia_isp *_handle;
};

class AiqAeHandler
    : public AeHandler
{
    friend class AiqCompositor;
private:
    struct AiqAeResult {
        ia_aiq_ae_results                 ae_result;
        ia_aiq_ae_exposure_result         ae_exp_ret;
        ia_aiq_exposure_parameters        aiq_exp_param;
        ia_aiq_exposure_sensor_parameters sensor_exp_param;
        ia_aiq_hist_weight_grid           weight_grid;
        ia_aiq_flash_parameters           flash_param;

        AiqAeResult();
        void copy (ia_aiq_ae_results *result);

        XCAM_DEAD_COPY (AiqAeResult);
    };

public:
    explicit AiqAeHandler (SmartPtr<AiqCompositor> &aiq_compositor);
    ~AiqAeHandler () {}

    bool is_started () const {
        return _started;
    }

    bool set_description (struct atomisp_sensor_mode_data *sensor_mode_data);

    ia_aiq_ae_results *get_result () {
        return &_result.ae_result;
    }

    //virtual functions from AnalyzerHandler
    virtual XCamReturn analyze (X3aResultList &output);

    // virtual functions from AeHandler
    virtual XCamFlickerMode get_flicker_mode ();
    virtual int64_t get_current_exposure_time ();
    virtual double get_current_analog_gain ();
    virtual double get_max_analog_gain ();

    XCamReturn set_RGBS_weight_grid (ia_aiq_rgbs_grid **out_rgbs_grid);
    XCamReturn set_hist_weight_grid (ia_aiq_hist_weight_grid **out_weight_grid);
    XCamReturn dump_hist_weight_grid (const ia_aiq_hist_weight_grid *weight_grid);
    XCamReturn dump_RGBS_grid (const ia_aiq_rgbs_grid *rgbs_grid);

private:
    bool ensure_ia_parameters ();
    bool ensure_ae_mode ();
    bool ensure_ae_metering_mode ();
    bool ensure_ae_priority_mode ();
    bool ensure_ae_flicker_mode ();
    bool ensure_ae_manual ();
    bool ensure_ae_ev_shift ();

    void adjust_ae_speed (
        ia_aiq_exposure_sensor_parameters &cur_res,
        const ia_aiq_exposure_sensor_parameters &last_res, double ae_speed);
    void adjust_ae_limitation (ia_aiq_exposure_sensor_parameters &cur_res);
    bool manual_control_result (
        ia_aiq_exposure_sensor_parameters &cur_res,
        const ia_aiq_exposure_sensor_parameters &last_res);

    SmartPtr<X3aResult> pop_result ();

    static void convert_xcam_window_to_ia (const XCam3AWindow &window, ia_rectangle &ia_window);

private:
    XCAM_DEAD_COPY (AiqAeHandler);

protected:
    SmartPtr<AiqCompositor>           _aiq_compositor;
    /* AIQ */
    ia_rectangle                      _ia_ae_window;
    ia_aiq_exposure_sensor_descriptor _sensor_descriptor;
    ia_aiq_ae_manual_limits           _manual_limits;

    ia_aiq_ae_input_params            _input;

    /* result */
    AiqAeResult                       _result;
    uint32_t                          _calculate_period;
    bool                              _started;
};

class AiqAwbHandler
    : public AwbHandler
{
    friend class AiqCompositor;
public:
    explicit AiqAwbHandler (SmartPtr<AiqCompositor> &aiq_compositor);
    ~AiqAwbHandler () {}

    virtual XCamReturn analyze (X3aResultList &output);

    ia_aiq_awb_results *get_result () {
        return &_result;
    }
    bool is_started () const {
        return _started;
    }

private:
    bool ensure_ia_parameters ();
    bool ensure_awb_mode ();
    void adjust_speed (const ia_aiq_awb_results &last_ret);

    XCAM_DEAD_COPY (AiqAwbHandler);

protected:
    SmartPtr<AiqCompositor>     _aiq_compositor;
    /*aiq*/
    ia_aiq_awb_input_params     _input;
    ia_aiq_awb_manual_cct_range _cct_range;

    ia_aiq_awb_results          _result;
    ia_aiq_awb_results          _history_result;
    bool                        _started;
};

class AiqAfHandler
    : public AfHandler
{
public:
    explicit AiqAfHandler (SmartPtr<AiqCompositor> &aiq_compositor)
        : _aiq_compositor (aiq_compositor)
    {}
    ~AiqAfHandler () {}

    virtual XCamReturn analyze (X3aResultList &output);

private:
    XCAM_DEAD_COPY (AiqAfHandler);

protected:
    SmartPtr<AiqCompositor>        _aiq_compositor;
};

class AiqCommonHandler
    : public CommonHandler
{
    friend class AiqCompositor;
public:
    explicit AiqCommonHandler (SmartPtr<AiqCompositor> &aiq_compositor);
    ~AiqCommonHandler () {}

    virtual XCamReturn analyze (X3aResultList &output);
    ia_aiq_gbce_results *get_gbce_result () {
        return _gbce_result;
    }
    XCamColorEffect get_color_effect() {
        return _params.color_effect;
    }

private:
    XCAM_DEAD_COPY (AiqCommonHandler);

protected:
    SmartPtr<AiqCompositor>     _aiq_compositor;
    ia_aiq_gbce_results        *_gbce_result;
};

class AiqCompositor {
public:
    explicit AiqCompositor ();
    ~AiqCompositor ();

    void set_ae_handler (SmartPtr<AiqAeHandler> &handler);
    void set_awb_handler (SmartPtr<AiqAwbHandler> &handler);
    void set_af_handler (SmartPtr<AiqAfHandler> &handler);
    void set_common_handler (SmartPtr<AiqCommonHandler> &handler);

    void set_frame_use (ia_aiq_frame_use value) {
        _frame_use = value;
    }
    void set_size (uint32_t width, uint32_t height) {
        _width = width;
        _height = height;
    }
    void get_size (uint32_t &out_width, uint32_t &out_height) const {
        out_width = _width;
        out_height = _height;
    }
    bool open (ia_binary_data &cpf);
    void close ();

    bool set_sensor_mode_data (struct atomisp_sensor_mode_data *sensor_mode);
    bool set_3a_stats (SmartPtr<X3aIspStatistics> &stats);

    ia_aiq  * get_handle () {
        return _ia_handle;
    }
    ia_aiq_frame_use get_frame_use () const {
        return _frame_use;
    }

    XCamReturn integrate (  X3aResultList &results);

    SmartPtr<X3aResult> generate_3a_configs (struct atomisp_parameters *parameters);
    void convert_window_to_ia (const XCam3AWindow &window, ia_rectangle &ia_window);
    XCamReturn convert_color_effect (IspInputParameters &isp_input);

    double get_ae_ev_shift_unlock () {
        return _ae_handler->get_ev_shift_unlock();
    }

private:
    XCamReturn apply_gamma_table (struct atomisp_parameters *isp_param);
    XCamReturn apply_night_mode (struct atomisp_parameters *isp_param);

    XCAM_DEAD_COPY (AiqCompositor);

private:
    SmartPtr<IaIspAdaptor>     _adaptor;
    SmartPtr<AiqAeHandler>     _ae_handler;
    SmartPtr<AiqAwbHandler>    _awb_handler;
    SmartPtr<AiqAfHandler>     _af_handler;
    SmartPtr<AiqCommonHandler> _common_handler;
    ia_aiq                    *_ia_handle;
    ia_mkn                    *_ia_mkn;
    ia_aiq_pa_results         *_pa_result;
    ia_aiq_frame_use           _frame_use;
    ia_aiq_frame_params        _frame_params;

    /*grids*/
    ;

    uint32_t                   _width;
    uint32_t                   _height;

};

};

#endif //XCAM_AIQ_HANDLER_H
