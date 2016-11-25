/*
 * smart_analyzer.h - smart analyzer
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */
#ifndef XCAM_SMART_ANALYZER_H
#define XCAM_SMART_ANALYZER_H

#include "xcam_analyzer.h"
#include "smart_analysis_handler.h"
#include "x3a_result_factory.h"

namespace XCam {

class BufferProxy;

class SmartAnalyzer
    : public XAnalyzer
{
public:
    SmartAnalyzer (const char *name = "SmartAnalyzer");
    ~SmartAnalyzer ();

    XCamReturn add_handler (SmartPtr<SmartAnalysisHandler> handler);
    XCamReturn update_params (XCamSmartAnalysisParam &params);
    void post_smart_results (X3aResultList &results, int64_t timestamp);

protected:
    virtual XCamReturn create_handlers ();
    virtual XCamReturn release_handlers ();
    virtual XCamReturn internal_init (uint32_t width, uint32_t height, double framerate);
    virtual XCamReturn internal_deinit ();
    virtual XCamReturn configure ();
    virtual XCamReturn analyze (SmartPtr<BufferProxy> &buffer);

private:
    XCAM_DEAD_COPY (SmartAnalyzer);

private:
    SmartHandlerList   _handlers;
    X3aResultList      _results;

    XCAM_OBJ_PROFILING_DEFINES;

};

}

#endif //XCAM_SMART_ANALYZER_H
