/*
 * pipe_manager.h - pipe manager
 *
 *  Copyright (c) 2016 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_PIPE_MANAGER_H
#define XCAM_PIPE_MANAGER_H

#include <xcam_std.h>
#include <smart_analyzer.h>
#include <x3a_image_process_center.h>
#include <stats_callback_interface.h>

namespace XCam {

class PipeManager
    : public StatsCallback
    , public AnalyzerCallback
    , public ImageProcessCallback
{
public:
    PipeManager ();
    virtual ~PipeManager ();

    bool set_smart_analyzer (SmartPtr<SmartAnalyzer> analyzer);
    bool add_image_processor (SmartPtr<ImageProcessor> processor);

    bool is_running () const {
        return _is_running;
    }

    XCamReturn start ();
    XCamReturn stop ();

    virtual XCamReturn push_buffer (SmartPtr<VideoBuffer> &buf);

protected:
    virtual void post_buffer (const SmartPtr<VideoBuffer> &buf) = 0;

    // virtual functions derived from PollCallback
    virtual XCamReturn scaled_image_ready (const SmartPtr<VideoBuffer> &buffer);

    // virtual functions derived from AnalyzerCallback
    virtual void x3a_calculation_done (XAnalyzer *analyzer, X3aResultList &results);
    virtual void x3a_calculation_failed (XAnalyzer *analyzer, int64_t timestamp, const char *msg);

    // virtual functions derived from ImageProcessCallback
    virtual void process_buffer_done (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_buffer_failed (ImageProcessor *processor, const SmartPtr<VideoBuffer> &buf);
    virtual void process_image_result_done (ImageProcessor *processor, const SmartPtr<X3aResult> &result);

private:
    XCAM_DEAD_COPY (PipeManager);

protected:
    bool                             _is_running;
    SmartPtr<SmartAnalyzer>          _smart_analyzer;
    SmartPtr<X3aImageProcessCenter>  _processor_center;
};

};

#endif // XCAM_PIPE_MANAGER_H
