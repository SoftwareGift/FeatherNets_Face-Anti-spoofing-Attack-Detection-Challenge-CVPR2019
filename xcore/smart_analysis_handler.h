/*
 * smart_analysis_handler.h - smart analysis handler
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
#ifndef XCAM_SMART_ANALYSIS_HANDLER_H
#define XCAM_SMART_ANALYSIS_HANDLER_H

#include <base/xcam_smart_description.h>
#include "smart_analyzer_loader.h"
#include "x3a_result_factory.h"

namespace XCam {

class BufferProxy;

class SmartAnalysisHandler
{
public:
    SmartAnalysisHandler (XCamSmartAnalysisDescription *desc, SmartPtr<SmartAnalyzerLoader> &loader, const char *name = "SmartHandler");
    ~SmartAnalysisHandler ();

    XCamReturn update_params (XCamSmartAnalysisParam &params);
    XCamReturn analyze (XCamVideoBuffer *buffer, X3aResultList &results);
    const char * get_name () const {
        return _name;
    }

protected:
    XCamReturn create_context ();
    void destroy_context ();

private:
    XCamReturn convert_results (XCam3aResultHead *from[], uint32_t from_count, X3aResultList &to);
    XCAM_DEAD_COPY (SmartAnalysisHandler);

private:
    XCamSmartAnalysisDescription *_desc;
    SmartPtr<SmartAnalyzerLoader> _loader;
    char *_name;
    XCamSmartAnalysisContext *_context;
};

}

#endif //XCAM_SMART_ANALYSIS_HANDLER_H
