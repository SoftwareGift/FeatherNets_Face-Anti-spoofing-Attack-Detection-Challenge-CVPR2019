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
 *             Wind Yuan <feng.yuan@intel.com>
 */
#ifndef XCAM_SMART_ANALYSIS_HANDLER_H
#define XCAM_SMART_ANALYSIS_HANDLER_H

#include <base/xcam_smart_description.h>
#include <map>
#include "x3a_result_factory.h"

namespace XCam {

class BufferProxy;
class SmartAnalysisHandler;
class SmartAnalyzerLoader;
class SmartAnalyzer;

typedef std::list<SmartPtr<SmartAnalysisHandler>> SmartHandlerList;

class SmartAnalysisHandler
{
    typedef std::map<XCamSmartAnalysisContext*, SmartPtr<SmartAnalysisHandler>> SmartHandlerMap;

public:
    SmartAnalysisHandler (XCamSmartAnalysisDescription *desc, SmartPtr<SmartAnalyzerLoader> &loader, const char *name = "SmartHandler");
    ~SmartAnalysisHandler ();
    void set_analyzer (SmartAnalyzer *analyzer) {
        _analyzer = analyzer;
    }

    XCamReturn create_context (SmartPtr<SmartAnalysisHandler> &self);
    void destroy_context ();
    bool is_valid () const {
        return (_context != NULL);
    }

    XCamReturn update_params (XCamSmartAnalysisParam &params);
    XCamReturn analyze (SmartPtr<BufferProxy> &buffer, X3aResultList &results);
    const char * get_name () const {
        return _name;
    }
    uint32_t get_priority () const {
        if (_desc)
            return _desc->priority;
        return 0;
    }

protected:
    XCamReturn post_smart_results (const XCamVideoBuffer *buffer, XCam3aResultHead *results[], uint32_t res_count);
    static XCamReturn post_aync_results (
        XCamSmartAnalysisContext *context,
        const XCamVideoBuffer *buffer,
        XCam3aResultHead *results[], uint32_t res_count);

private:
    XCamReturn convert_results (XCam3aResultHead *from[], uint32_t from_count, X3aResultList &to);
    XCAM_DEAD_COPY (SmartAnalysisHandler);

//
private:
    static SmartHandlerMap          _handler_map;
    static Mutex                    _handler_map_lock;

private:
    XCamSmartAnalysisDescription   *_desc;
    SmartPtr<SmartAnalyzerLoader>   _loader;
    SmartAnalyzer                  *_analyzer;
    char                           *_name;
    XCamSmartAnalysisContext       *_context;
    bool                            _async_mode;
};

}

#endif //XCAM_SMART_ANALYSIS_HANDLER_H
