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

#include "xcam_common.h"
#include "xcam_defs.h"
#include "xcam_mutex.h"
#include "xcam_utils.h"
#include "x3a_result.h"
#include "xcam_3a_types.h"

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
    XCamAeMode get_mode_unlock() const {
        return _mode;
    }
    XCamAeMeteringMode get_metering_mode_unlock() const {
        return _metering_mode;
    }
    const XCam3AWindow &get_window_unlock() const {
        return _window;
    }
    XCamFlickerMode get_flicker_mode_unlock() const {
        return _flicker_mode;
    }
    double get_speed_unlock() const {
        return _speed;
    }
    double get_ev_shift_unlock() const {
        return _ev_shift;
    }

private:
    XCAM_DEAD_COPY (AeHandler);

protected:
    XCamAeMode              _mode;
    XCamAeMeteringMode      _metering_mode;
    XCam3AWindow            _window;
    XCamFlickerMode         _flicker_mode;
    double                  _speed;

    /* exposure limitation */
    uint64_t                _exposure_time_min, _exposure_time_max; // exposure time range
    double                  _max_analog_gain;

    /* exposure manual values */
    uint64_t                _manual_exposure_time;
    double                  _manual_analog_gain;

    double                  _aperture_fn;

    /*ev*/
    double                  _ev_shift;
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
    XCamAwbMode get_mode_unlock() const {
        return _mode;
    }
    double get_speed_unlock () const {
        return _speed;
    }

private:
    XCAM_DEAD_COPY (AwbHandler);

protected:
    XCamAwbMode             _mode;
    double                  _speed;
    uint32_t                _cct_min, _cct_max;
    XCam3AWindow            _window;

    /* manual gain */
    double                  _gr_gain;
    double                  _r_gain;
    double                  _b_gain;
    double                  _gb_gain;
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

public:
    bool has_gbce_unlock () const {
        return _enable_gbce;
    }
    bool has_dvs_unlock () const {
        return _enable_dvs;
    }
    bool has_night_mode_unlock () const {
        return _enable_night_mode;
    }

    double get_nr_level_unlock () const {
        return _nr_level;
    }
    double get_tnr_level_unlock () const {
        return _tnr_level;
    }
    double get_brightness_unlock () const {
        return _brightness;
    }
    double get_contrast_unlock () const {
        return _contrast;
    }
    double get_hue_unlock () const {
        return _hue;
    }
    double get_saturation_unlock () const {
        return _saturation;
    }
    double get_sharpness_unlock () const {
        return _sharpness;
    }

private:
    XCAM_DEAD_COPY (CommonHandler);

protected:
    /* R, G, B gamma table, size = XCAM_GAMMA_TABLE_SIZE */
    bool                      _is_manual_gamma;
    double                    _r_gamma [XCAM_GAMMA_TABLE_SIZE];
    double                    _g_gamma [XCAM_GAMMA_TABLE_SIZE];
    double                    _b_gamma [XCAM_GAMMA_TABLE_SIZE];

    /*
     * manual brightness, contrast, hue, saturation, sharpness
     * -1.0 < value < 1.0
     */
    double                     _nr_level;
    double                     _tnr_level;

    double                     _brightness;
    double                     _contrast;
    double                     _hue;
    double                     _saturation;
    double                     _sharpness;

    /* others */
    bool                       _enable_dvs;
    bool                       _enable_gbce;
    bool                       _enable_night_mode;
};

};

#endif // XCAM_HANDLER_INTERFACE_H
