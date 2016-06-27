/*
 * dynamic_analyzer_loader.h - dynamic analyzer loader
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 *         Zong Wei  <wei.zong@intel.com>
 */

#ifndef XCAM_DYNAMIC_ANALYZER_LOADER_H
#define XCAM_DYNAMIC_ANALYZER_LOADER_H

#include <base/xcam_common.h>
#include <base/xcam_3a_description.h>
#include "analyzer_loader.h"
#include "smartptr.h"

namespace XCam {
class X3aAnalyzer;

class DynamicAnalyzerLoader
    : public AnalyzerLoader
{
public:
    DynamicAnalyzerLoader (const char *lib_path, const char *symbol = XCAM_3A_LIB_DESCRIPTION);
    virtual ~DynamicAnalyzerLoader ();

    virtual SmartPtr<X3aAnalyzer> load_analyzer (SmartPtr<AnalyzerLoader> &self);

protected:
    virtual void *load_symbol (void* handle);

private:
    XCAM_DEAD_COPY(DynamicAnalyzerLoader);
};

};

#endif // XCAM_DYNAMIC_ANALYZER_LOADER_H