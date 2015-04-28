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

#include "x3a_analyzer.h"
#include "xcam_thread.h"
#include "safe_list.h"
#include "x3a_stats_pool.h"

namespace XCam {

class AnalyzerThread
    : public Thread
{
public:
    AnalyzerThread (X3aAnalyzer *analyzer);
    ~AnalyzerThread ();

    void triger_stop() {
        _3a_stats_queue.wakeup ();
    }
    bool push_stats (const SmartPtr<X3aStats> &stats);

protected:
    virtual bool started ();
    virtual void stopped () {
        _3a_stats_queue.clear ();
    }
    virtual bool loop ();

private:
    X3aAnalyzer               *_analyzer;
    SafeList<X3aStats> _3a_stats_queue;
};

AnalyzerThread::AnalyzerThread (X3aAnalyzer *analyzer)
    : Thread ("AnalyzerThread")
    , _analyzer (analyzer)
{}

AnalyzerThread::~AnalyzerThread ()
{
    _3a_stats_queue.clear ();
}

bool
AnalyzerThread::push_stats (const SmartPtr<X3aStats> &stats)
{
    _3a_stats_queue.push (stats);
    return true;
}

bool
AnalyzerThread::started ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_analyzer);
    ret = _analyzer->configure_3a ();
    if (ret != XCAM_RETURN_NO_ERROR) {
        _analyzer->notify_calculation_failed (NULL, 0, "configure 3a failed");
        XCAM_LOG_WARNING ("analyzer(%s) configure 3a failed", XCAM_STR(_analyzer->get_name()));
        return false;
    }

    return true;
}

bool
AnalyzerThread::loop ()
{
    const static int32_t timeout = -1;
    SmartPtr<X3aStats> stats = _3a_stats_queue.pop (timeout);
    if (!stats.ptr()) {
        XCAM_LOG_DEBUG ("analyzer thread got empty stats, stop thread");
        return false;
    }
    XCamReturn ret = _analyzer->analyze_3a_statistics (stats);
    if (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS)
        return true;

    XCAM_LOG_DEBUG ("analyzer(%s) failed to analyze 3a stats", XCAM_STR(_analyzer->get_name()));
    return false;
}

void
AnalyzerCallback::x3a_calculation_done (X3aAnalyzer *analyzer, X3aResultList &results)
{
    XCAM_UNUSED (analyzer);

    for (X3aResultList::iterator i_res = results.begin();
            i_res != results.end(); ++i_res) {
        SmartPtr<X3aResult> res = *i_res;
        XCAM_LOG_DEBUG (
            "calculated 3a result(type:%d, timestamp:" XCAM_TIMESTAMP_FORMAT ")",
            res->get_type (), XCAM_TIMESTAMP_ARGS (res->get_timestamp ()));
    }
}

void
AnalyzerCallback::x3a_calculation_failed (X3aAnalyzer *analyzer, int64_t timestamp, const char *msg)
{
    XCAM_UNUSED (analyzer);

    XCAM_LOG_WARNING (
        "Calculate 3a result failed, ts(" XCAM_TIMESTAMP_FORMAT "), msg:%s",
        XCAM_TIMESTAMP_ARGS (timestamp), XCAM_STR (msg));
}

X3aAnalyzer::X3aAnalyzer (const char *name)
    : _name (NULL)
    , _width (0)
    , _height (0)
    , _framerate (30.0)
    , _ae_handler (NULL)
    , _awb_handler (NULL)
    , _af_handler (NULL)
    , _common_handler (NULL)
    , _callback (NULL)
{
    if (name)
        _name = strdup (name);
    _3a_analyzer_thread  = new AnalyzerThread (this);
}

X3aAnalyzer::~X3aAnalyzer()
{
    if (_name)
        xcam_free (_name);
}

bool
X3aAnalyzer::set_results_callback (AnalyzerCallback *callback)
{
    XCAM_ASSERT (!_callback);
    _callback = callback;
    return true;
}

XCamReturn
X3aAnalyzer::init (uint32_t width, uint32_t height, double framerate)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<AeHandler> ae_handler;
    SmartPtr<AwbHandler> awb_handler;
    SmartPtr<AfHandler> af_handler;
    SmartPtr<CommonHandler> common_handler;


    XCAM_ASSERT (!_width && !_height);
    _width = width;
    _height = height;
    _framerate = framerate;

    XCAM_ASSERT (!_ae_handler.ptr() || !_awb_handler.ptr() ||
                 !_af_handler.ptr() || !_common_handler.ptr());

    ae_handler = create_ae_handler ();
    awb_handler = create_awb_handler ();
    af_handler = create_af_handler ();
    common_handler = create_common_handler ();

    if (!ae_handler.ptr() || !awb_handler.ptr() || !af_handler.ptr() || !common_handler.ptr()) {
        XCAM_LOG_WARNING ("create handlers failed");
        deinit ();
        return XCAM_RETURN_ERROR_MEM;
    }
    _ae_handler = ae_handler;
    _awb_handler = awb_handler;
    _af_handler = af_handler;
    _common_handler = common_handler;

    ret = internal_init (width, height, _framerate);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_WARNING ("analyzer init failed");
        deinit ();
        return ret;
    }

    XCAM_LOG_INFO (
        "Analyzer(%s) initialized(w:%d, h:%d).",
        XCAM_STR(get_name()), _width, _height);
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzer::deinit ()
{
    internal_deinit();

    _ae_handler.release ();
    _awb_handler.release ();
    _af_handler.release ();
    _common_handler.release ();

    _width = 0;
    _height = 0;

    XCAM_LOG_INFO ("Analyzer(%s) deinited.", XCAM_STR(get_name()));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzer::start ()
{
    if (_3a_analyzer_thread->start () == false) {
        XCAM_LOG_WARNING ("analyzer thread start failed");
        stop ();
        return XCAM_RETURN_ERROR_THREAD;
    }

    XCAM_LOG_INFO ("Analyzer(%s) started.", XCAM_STR(get_name()));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzer::stop ()
{
    _3a_analyzer_thread->triger_stop ();
    _3a_analyzer_thread->stop ();

    XCAM_LOG_INFO ("Analyzer(%s) stopped.", XCAM_STR(get_name()));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
X3aAnalyzer::push_3a_stats (const SmartPtr<X3aStats> &stats)
{
    if (!_3a_analyzer_thread->is_running())
        return XCAM_RETURN_ERROR_THREAD;

    if (_3a_analyzer_thread->push_stats (stats))
        return XCAM_RETURN_NO_ERROR;

    return XCAM_RETURN_ERROR_THREAD;
}

XCamReturn
X3aAnalyzer::analyze_3a_statistics (SmartPtr<X3aStats> &stats)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    X3aResultList results;

    ret = pre_3a_analyze (stats);
    if (ret != XCAM_RETURN_NO_ERROR) {
        notify_calculation_failed(
            NULL, stats->get_timestamp (), "ae calculation failed");
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

    if (!results.empty ())
        notify_calculation_done (results);

    return ret;

}

void
X3aAnalyzer::notify_calculation_failed (AnalyzerHandler *handler, int64_t timestamp, const char *msg)
{
    XCAM_UNUSED (handler);

    if (_callback)
        _callback->x3a_calculation_failed (this, timestamp, msg);
    XCAM_LOG_DEBUG (
        "calculation failed on ts:" XCAM_TIMESTAMP_FORMAT ", reason:%s",
        XCAM_TIMESTAMP_ARGS (timestamp), XCAM_STR (msg));
}

void
X3aAnalyzer::notify_calculation_done (X3aResultList &results)
{
    XCAM_ASSERT (!results.empty ());
    if (_callback)
        _callback->x3a_calculation_done (this, results);
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

};
