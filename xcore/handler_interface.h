/*
 * handler_interface.h - handler interface
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

#ifndef XCAM_HANDLER_INTERFACE_H
#define XCAM_HANDLER_INTERFACE_H

#include <base/xcam_common.h>
#include <base/xcam_3a_types.h>
#include <base/xcam_params.h>

#include "xcam_utils.h"
#include "xcam_mutex.h"
#include "x3a_result.h"

namespace XCam {

class AnalyzerHandler {
    friend class HanlderLock;
public:
    explicit AnalyzerHandler() {}
    virtual ~AnalyzerHandler () {}

    virtual XCamReturn analyze (X3aResultList &output) = 0;

protected:
    class HanlderLock
        : public SmartLock
    {
    public:
        HanlderLock(AnalyzerHandler *handler)
            : SmartLock (handler->_mutex)
        {}
        ~HanlderLock() {}
    };

    // members
    Mutex _mutex;
};

class AeHandler
    : public AnalyzerHandler
{
public:
    explicit AeHandler();
    virtual ~AeHandler() {}

    bool set_mode (XCamAeMode mode);
    bool set_metering_mode (XCamAeMeteringMode mode);
    bool set_window (XCam3AWindow *window);
    bool set_ev_shift (double ev_shift);
    bool set_speed (double speed);
    bool set_flicker_mode (XCamFlickerMode flicker);
    bool set_manual_exposure_time (int64_t time_in_us);
    bool set_manual_analog_gain (double gain);
    bool set_aperture (double fn);
    bool set_max_analog_gain (double max_gain);
    bool set_exposure_time_range (int64_t min_time_in_us, int64_t max_time_in_us);
    bool get_exposure_time_range (int64_t *min_time_in_us, int64_t *max_time_in_us);

    //virtual functions
    virtual XCamFlickerMode get_flicker_mode ();
    virtual int64_t get_current_exposure_time ();
    virtual double get_current_analog_gain ();
    virtual double get_max_analog_gain ();

protected:
    const XCamAeParam &get_params_unlock () const {
        return _params;
    }

    XCamAeMode get_mode_unlock() const {
        return _params.mode;
    }
    XCamAeMeteringMode get_metering_mode_unlock() const {
        return _params.metering_mode;
    }
    const XCam3AWindow &get_window_unlock() const {
        return _params.window;
    }
    XCamFlickerMode get_flicker_mode_unlock() const {
        return _params.flicker_mode;
    }
    double get_speed_unlock() const {
        return _params.speed;
    }
    double get_ev_shift_unlock() const {
        return _params.ev_shift;
    }

    uint64_t get_manual_exposure_time_unlock () const {
        return _params.manual_exposure_time;
    }
    double get_manual_analog_gain_unlock () const {
        return _params.manual_analog_gain;
    }

    double get_aperture_fn_unlock () const {
        return _params.aperture_fn;
    }

    void get_exposure_time_range_unlock (uint64_t &min, uint64_t &max) const {
        min = _params.exposure_time_min;
        max = _params.exposure_time_max;
    }

    double get_max_analog_gain_unlock () const {
        return _params.max_analog_gain;
    }

private:
    void reset_parameters ();
    XCAM_DEAD_COPY (AeHandler);

protected:
    XCamAeParam   _params;
};

class AwbHandler
    : public AnalyzerHandler
{
public:
    explicit AwbHandler();
    virtual ~AwbHandler() {}

    bool set_mode (XCamAwbMode mode);
    bool set_speed (double speed);
    bool set_color_temperature_range (uint32_t cct_min, uint32_t cct_max);
    bool set_manual_gain (double gr, double r, double b, double gb);

protected:
    const XCamAwbParam &get_params_unlock () const {
        return _params;
    }

    XCamAwbMode get_mode_unlock() const {
        return _params.mode;
    }
    double get_speed_unlock () const {
        return _params.speed;
    }

    const XCam3AWindow &get_window_unlock () const {
        return _params.window;
    }

    void get_cct_range_unlock (uint32_t &cct_min, uint32_t &cct_max) const {
        cct_min = _params.cct_min;
        cct_max = _params.cct_max;
    }

private:
    void reset_parameters ();
    XCAM_DEAD_COPY (AwbHandler);

protected:
    XCamAwbParam _params;
};

class AfHandler
    : public AnalyzerHandler
{
public:
    explicit AfHandler() {}
    virtual ~AfHandler() {}
private:
    XCAM_DEAD_COPY (AfHandler);

protected:
    const XCamAfParam &get_params_unlock () const {
        return _params;
    }

protected:
    XCamAfParam _params;
};

class CommonHandler
    : public AnalyzerHandler
{
public:
    explicit CommonHandler();
    virtual ~CommonHandler() {}

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

protected:
    const XCamCommonParam &get_params_unlock () const {
        return _params;
    }
    bool has_gbce_unlock () const {
        return _params.enable_gbce;
    }
    bool has_dvs_unlock () const {
        return _params.enable_dvs;
    }
    bool has_night_mode_unlock () const {
        return _params.enable_night_mode;
    }

    double get_nr_level_unlock () const {
        return _params.nr_level;
    }
    double get_tnr_level_unlock () const {
        return _params.tnr_level;
    }
    double get_brightness_unlock () const {
        return _params.brightness;
    }
    double get_contrast_unlock () const {
        return _params.contrast;
    }
    double get_hue_unlock () const {
        return _params.hue;
    }
    double get_saturation_unlock () const {
        return _params.saturation;
    }
    double get_sharpness_unlock () const {
        return _params.sharpness;
    }

private:
    void reset_parameters ();
    XCAM_DEAD_COPY (CommonHandler);

protected:
    XCamCommonParam _params;
};

};

#endif // XCAM_HANDLER_INTERFACE_H
