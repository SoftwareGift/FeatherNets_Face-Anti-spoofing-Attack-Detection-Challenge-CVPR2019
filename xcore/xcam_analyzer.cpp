/*
 * xcam_analyzer.cpp - libxcam analyzer
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
 *         Zong Wei <wei.zong@intel.com>
 *         Jia Meng <jia.meng@intel.com>
 */

#include "xcam_analyzer.h"
#include "x3a_stats_pool.h"

namespace XCam {

AnalyzerThread::AnalyzerThread (XAnalyzer *analyzer)
    : Thread ("AnalyzerThread")
    , _analyzer (analyzer)
{}

AnalyzerThread::~AnalyzerThread ()
{
    _stats_queue.clear ();
}

bool
AnalyzerThread::push_stats (const SmartPtr<BufferProxy> &buffer)
{
    _stats_queue.push (buffer);
    return true;
}

bool
AnalyzerThread::started ()
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (_analyzer);
    ret = _analyzer->configure ();
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
    SmartPtr<BufferProxy> latest_stats;
    SmartPtr<BufferProxy> stats = _stats_queue.pop (timeout);
    if (!stats.ptr()) {
        XCAM_LOG_DEBUG ("analyzer thread got empty stats, stop thread");
        return false;
    }
    //while ((latest_stats = _stats_queue.pop (0)).ptr ()) {
    //    stats = latest_stats;
    //    XCAM_LOG_WARNING ("lost 3a stats since 3a analyzer too slow");
    //}

    XCamReturn ret = _analyzer->analyze (stats);
    if (ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS)
        return true;

    XCAM_LOG_DEBUG ("analyzer(%s) failed to analyze 3a stats", XCAM_STR(_analyzer->get_name()));
    return false;
}

void
AnalyzerCallback::x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results)
{
    XCAM_UNUSED (analyzer);

    for (X3aResultList::iterator i_res = results.begin();
            i_res != results.end(); ++i_res) {
        SmartPtr<X3aResult> res = *i_res;
        if (res.ptr() == NULL) continue;
        XCAM_LOG_DEBUG (
            "calculated 3a result(type:0x%x, timestamp:" XCAM_TIMESTAMP_FORMAT ")",
            res->get_type (), XCAM_TIMESTAMP_ARGS (res->get_timestamp ()));
    }
}

void
AnalyzerCallback::x3a_calculation_failed (XAnalyzer *analyzer, int64_t timestamp, const char *msg)
{
    XCAM_UNUSED (analyzer);

    XCAM_LOG_WARNING (
        "Calculate 3a result failed, ts(" XCAM_TIMESTAMP_FORMAT "), msg:%s",
        XCAM_TIMESTAMP_ARGS (timestamp), XCAM_STR (msg));
}

XAnalyzer::XAnalyzer (const char *name)
    : _name (NULL)
    , _sync (false)
    , _started (false)
    , _width (0)
    , _height (0)
    , _framerate (30.0)
    , _callback (NULL)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    _analyzer_thread  = new AnalyzerThread (this);
}

XAnalyzer::~XAnalyzer()
{
    if (_name)
        xcam_free (_name);
}

bool
XAnalyzer::set_results_callback (AnalyzerCallback *callback)
{
    XCAM_ASSERT (!_callback);
    _callback = callback;
    return true;
}

XCamReturn
XAnalyzer::prepare_handlers ()
{
    return create_handlers ();
}

XCamReturn
XAnalyzer::init (uint32_t width, uint32_t height, double framerate)
{
    XCAM_LOG_DEBUG ("Analyzer(%s) init.", XCAM_STR(get_name()));
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (!_width && !_height);
    _width = width;
    _height = height;
    _framerate = framerate;

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
XAnalyzer::deinit ()
{
    internal_deinit ();

    release_handlers ();

    _width = 0;
    _height = 0;

    XCAM_LOG_INFO ("Analyzer(%s) deinited.", XCAM_STR(get_name()));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
XAnalyzer::set_sync_mode (bool sync)
{
    if (_started) {
        XCAM_LOG_ERROR ("can't set_sync_mode after analyzer started");
        return XCAM_RETURN_ERROR_PARAM;
    }
    _sync = sync;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
XAnalyzer::start ()
{
    if (_sync) {
        XCamReturn ret = configure ();
        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("analyzer failed to start in sync mode");
            stop ();
            return ret;
        }
    } else {
        if (_analyzer_thread->start () == false) {
            XCAM_LOG_WARNING ("analyzer thread start failed");
            stop ();
            return XCAM_RETURN_ERROR_THREAD;
        }
    }

    _started = true;
    XCAM_LOG_INFO ("Analyzer(%s) started in %s mode.", XCAM_STR(get_name()),
                   _sync ? "sync" : "async");
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
XAnalyzer::stop ()
{
    if (!_sync) {
        _analyzer_thread->triger_stop ();
        _analyzer_thread->stop ();
    }

    _started = false;
    XCAM_LOG_INFO ("Analyzer(%s) stopped.", XCAM_STR(get_name()));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
XAnalyzer::push_buffer (const SmartPtr<BufferProxy> &buffer)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    if (get_sync_mode ()) {
        SmartPtr<BufferProxy> data = buffer;
        ret = analyze (data);
    }
    else {
        if (!_analyzer_thread->is_running())
            return XCAM_RETURN_ERROR_THREAD;

        if (!_analyzer_thread->push_stats (buffer))
            return XCAM_RETURN_ERROR_THREAD;
    }

    return ret;
}

void
XAnalyzer::set_results_timestamp (X3aResultList &results, int64_t timestamp)
{
    if (results.empty ())
        return;

    X3aResultList::iterator i_results = results.begin ();
    for (; i_results != results.end ();  ++i_results)
    {
        (*i_results)->set_timestamp(timestamp);
    }
}

void
XAnalyzer::notify_calculation_failed (AnalyzerHandler *handler, int64_t timestamp, const char *msg)
{
    XCAM_UNUSED (handler);

    if (_callback)
        _callback->x3a_calculation_failed (this, timestamp, msg);
    XCAM_LOG_DEBUG (
        "calculation failed on ts:" XCAM_TIMESTAMP_FORMAT ", reason:%s",
        XCAM_TIMESTAMP_ARGS (timestamp), XCAM_STR (msg));
}

void
XAnalyzer::notify_calculation_done (X3aResultList &results)
{
    XCAM_ASSERT (!results.empty ());
    if (_callback)
        _callback->x3a_calculation_done (this, results);
}

};
