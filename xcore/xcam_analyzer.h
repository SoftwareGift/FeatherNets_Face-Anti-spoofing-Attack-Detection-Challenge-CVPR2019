/*
 * xcam_analyzer.h - libxcam analyzer
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
 */

#ifndef XCAM_ANALYZER_H
#define XCAM_ANALYZER_H

#include "xcam_utils.h"
#include "handler_interface.h"
#include "xcam_thread.h"
#include "buffer_pool.h"

namespace XCam {

class XAnalyzer;

class AnalyzerThread
    : public Thread
{
public:
    AnalyzerThread (XAnalyzer *analyzer);
    ~AnalyzerThread ();

    void triger_stop() {
        _stats_queue.pause_pop ();
    }
    bool push_stats (const SmartPtr<BufferProxy> &buffer);

protected:
    virtual bool started ();
    virtual void stopped () {
        _stats_queue.clear ();
    }
    virtual bool loop ();

private:
    XAnalyzer              *_analyzer;
    SafeList<BufferProxy>   _stats_queue;
};

class AnalyzerCallback {
public:
    explicit AnalyzerCallback () {}
    virtual ~AnalyzerCallback () {}
    virtual void x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results);
    virtual void x3a_calculation_failed (XAnalyzer *analyzer, int64_t timestamp, const char *msg);

private:
    XCAM_DEAD_COPY (AnalyzerCallback);
};

class AnalyzerThread;

class XAnalyzer {
    friend class AnalyzerThread;
public:
    explicit XAnalyzer (const char *name = NULL);
    virtual ~XAnalyzer ();

    bool set_results_callback (AnalyzerCallback *callback);
    XCamReturn prepare_handlers ();

    // prepare_handlers must called before init
    XCamReturn init (uint32_t width, uint32_t height, double framerate);
    XCamReturn deinit ();
    // set_sync_mode must be called before start
    XCamReturn set_sync_mode (bool sync);
    bool get_sync_mode () const {
        return _sync;
    };
    XCamReturn start ();
    XCamReturn stop ();
    XCamReturn push_buffer (const SmartPtr<BufferProxy> &buffer);

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

protected:
    /* virtual function list */
    virtual XCamReturn create_handlers () = 0;
    virtual XCamReturn release_handlers () = 0;
    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate) = 0;
    virtual XCamReturn internal_deinit () = 0;

    // in analyzer thread
    virtual XCamReturn configure () = 0;
    virtual XCamReturn analyze (SmartPtr<BufferProxy> &buffer) = 0;

protected:
    void notify_calculation_done (X3aResultList &results);
    void notify_calculation_failed (AnalyzerHandler *handler, int64_t timestamp, const char *msg);
    void set_results_timestamp (X3aResultList &results, int64_t timestamp);

private:

    XCAM_DEAD_COPY (XAnalyzer);

protected:
    SmartPtr<AnalyzerThread> _analyzer_thread;

private:
    char                    *_name;
    bool                     _sync;
    bool                     _started;
    uint32_t                 _width;
    uint32_t                 _height;
    double                   _framerate;
    AnalyzerCallback        *_callback;
};

}
#endif //XCAM_ANALYZER_H
