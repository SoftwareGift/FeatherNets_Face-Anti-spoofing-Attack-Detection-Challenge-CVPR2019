/*
 * smart_analyzer_loader.h - smart analyzer loader
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

#ifndef XCAM_SMART_ANALYZER_LOADER_H
#define XCAM_SMART_ANALYZER_LOADER_H

#include "xcam_utils.h"
#include "smartptr.h"
#include "analyzer_loader.h"
#include "smart_analysis_handler.h"
#include <base/xcam_smart_description.h>
#include <list>

namespace XCam {

class SmartAnalyzer;
class SmartAnalysisHandler;
class SmartAnalyzerLoader;

typedef std::list<SmartPtr<SmartAnalyzerLoader>> AnalyzerLoaderList;

class SmartAnalyzerLoader
    : public AnalyzerLoader
{

public:
    SmartAnalyzerLoader (const char *lib_path, const char *name = NULL, const char *symbol = XCAM_SMART_ANALYSIS_LIB_DESCRIPTION);
    virtual ~SmartAnalyzerLoader ();

    static SmartHandlerList load_smart_handlers (const char *dir_path);

protected:
    static AnalyzerLoaderList create_analyzer_loader (const char *dir_path);
    SmartPtr<SmartAnalysisHandler> load_smart_handler (SmartPtr<SmartAnalyzerLoader> &self);

protected:
    virtual void *load_symbol (void* handle);

private:
    XCAM_DEAD_COPY (SmartAnalyzerLoader);

private:
    char *_name;
};

};
#endif //XCAM_SMART_ANALYZER_LOADER_H
