/*
 * x3a_analyzer.h - 3a analyzer
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

#ifndef XCAM_3A_ANALYZER_H
#define XCAM_3A_ANALYZER_H

#include "xcam_utils.h"
#include "handler_interface.h"

namespace XCam {

class X3aAnalyzer;
class X3aStats;

class AnalyzerCallback {
public:
    explicit AnalyzerCallback () {}
    virtual ~AnalyzerCallback () {}
    virtual void x3a_calculation_done (X3aAnalyzer *analyzer, X3aResultList &results);
    virtual void x3a_calculation_failed (X3aAnalyzer *analyzer, int64_t timestamp, const char *msg);

private:
    XCAM_DEAD_COPY (AnalyzerCallback);
};

class AnalyzerThread;

class X3aAnalyzer {
    friend class AnalyzerThread;
public:
    explicit X3aAnalyzer (const char *name = NULL);
    virtual ~X3aAnalyzer ();

    bool set_results_callback (AnalyzerCallback *callback);

    XCamReturn prepare_handlers ();
    // prepare_handlers must called before init
    XCamReturn init (uint32_t width, uint32_t height, double framerate);
    XCamReturn deinit ();
    // set_sync_mode must be called before start
    XCamReturn set_sync_mode (bool sync);
    XCamReturn start ();
    XCamReturn stop ();

    /* analyze 3A statistics */
    XCamReturn push_3a_stats (const SmartPtr<X3aStats> &stats);

    /* AWB */
    bool set_awb_mode (XCamAwbMode mode);
    bool set_awb_speed (double speed);
    bool set_awb_color_temperature_range (uint32_t cct_min, uint32_t cct_max);
    bool set_awb_manual_gain (double gr, double r, double b, double gb);

    /* AE */
    bool set_ae_mode (XCamAeMode mode);
    bool set_ae_metering_mode (XCamAeMeteringMode mode);
    bool set_ae_window (XCam3AWindow *window, uint8_t count = 1);
    bool set_ae_ev_shift (double ev_shift);
    bool set_ae_speed (double speed);
    bool set_ae_flicker_mode (XCamFlickerMode flicker);

    XCamFlickerMode get_ae_flicker_mode ();
    uint64_t get_ae_current_exposure_time ();
    double get_ae_current_analog_gain ();

    bool set_ae_manual_exposure_time (int64_t time_in_us);
    bool set_ae_manual_analog_gain (double gain);
    bool set_ae_aperture (double fn);
    bool set_ae_max_analog_gain (double max_gain);
    double get_ae_max_analog_gain ();
    bool set_ae_exposure_time_range (int64_t min_time_in_us, int64_t max_time_in_us);
    bool get_ae_exposure_time_range (int64_t *min_time_in_us, int64_t *max_time_in_us);

    /* DVS */
    bool set_dvs (bool enable);
    bool set_gbce (bool enable);
    bool set_night_mode (bool enable);

    /* Picture quality */
    bool set_noise_reduction_level (double level);
    bool set_temporal_noise_reduction_level (double level);
    bool set_manual_brightness (double level);
    bool set_manual_contrast (double level);
    bool set_manual_hue (double level);
    bool set_manual_saturation (double level);
    bool set_manual_sharpness (double level);
    bool set_gamma_table (double *r_table, double *g_table, double *b_table);
    bool set_color_effect(XCamColorEffect effect);
    bool set_parameter_brightness (double level);

    // whole update of parameters
    bool update_awb_parameters (const XCamAwbParam &params);
    bool update_ae_parameters (const XCamAeParam &params);
    bool update_af_parameters (const XCamAfParam &params);
    bool update_common_parameters (const XCamCommonParam &params);

    uint32_t get_width () const {
        return _width;
    }
    uint32_t get_height () const {
        return _height;
    }

    double get_framerate () const {
        return _framerate;
    }
    const char * get_name () const {
        return _name;
    }

    SmartPtr<AeHandler> get_ae_handler () {
        return _ae_handler;
    }
    SmartPtr<AwbHandler> get_awb_handler () {
        return _awb_handler;
    }
    SmartPtr<AfHandler> get_af_handler () {
        return _af_handler;
    }
    SmartPtr<CommonHandler> get_common_handler () {
        return _common_handler;
    }

protected:
    /* virtual function list */
    virtual SmartPtr<AeHandler> create_ae_handler () = 0;
    virtual SmartPtr<AwbHandler> create_awb_handler () = 0;
    virtual SmartPtr<AfHandler> create_af_handler () = 0;
    virtual SmartPtr<CommonHandler> create_common_handler () = 0;
    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate) = 0;
    virtual XCamReturn internal_deinit () = 0;

    // in 3a stats thread
    virtual XCamReturn configure_3a () = 0;
    // @param[in]   stats,  3a statistics prepared
    virtual XCamReturn pre_3a_analyze (SmartPtr<X3aStats> &stats) = 0;
    // @param[out]  results,   new 3a results merged into \c results
    virtual XCamReturn post_3a_analyze (X3aResultList &results) = 0;

protected:
    void notify_calculation_done (X3aResultList &results);
    void notify_calculation_failed (AnalyzerHandler *handler, int64_t timestamp, const char *msg);

private:
    XCamReturn analyze_3a_statistics (SmartPtr<X3aStats> &stats);

    XCAM_DEAD_COPY (X3aAnalyzer);

protected:
    double                   _brightness_level_param;

private:
    char                    *_name;
    bool                     _sync;
    bool                     _started;
    uint32_t                 _width;
    uint32_t                 _height;
    double                   _framerate;

    SmartPtr<AeHandler>      _ae_handler;
    SmartPtr<AwbHandler>     _awb_handler;
    SmartPtr<AfHandler>      _af_handler;
    SmartPtr<CommonHandler>  _common_handler;

    SmartPtr<AnalyzerThread> _3a_analyzer_thread;
    AnalyzerCallback        *_callback;

};

}
#endif //XCAM_3A_ANALYZER_H
