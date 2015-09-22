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
#include "xcam_analyzer.h"
#include "handler_interface.h"

namespace XCam {

class X3aStats;
class AnalyzerThread;
class BufferProxy;

class X3aAnalyzer
    : public XAnalyzer
{
    friend class AnalyzerThread;
public:
    explicit X3aAnalyzer (const char *name = NULL);
    virtual ~X3aAnalyzer ();

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
    virtual XCamReturn create_handlers ();
    virtual XCamReturn release_handlers ();
    virtual XCamReturn configure ();
    virtual XCamReturn analyze (SmartPtr<BufferProxy> &buffer);

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

private:
    XCamReturn analyze_3a_statistics (SmartPtr<X3aStats> &stats);

    XCAM_DEAD_COPY (X3aAnalyzer);

protected:
    double                   _brightness_level_param;

private:
    SmartPtr<AeHandler>      _ae_handler;
    SmartPtr<AwbHandler>     _awb_handler;
    SmartPtr<AfHandler>      _af_handler;
    SmartPtr<CommonHandler>  _common_handler;
};

}
#endif //XCAM_3A_ANALYZER_H
