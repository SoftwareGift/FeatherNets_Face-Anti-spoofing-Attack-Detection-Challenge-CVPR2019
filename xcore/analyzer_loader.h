/*
 * analyzer_loader.h - analyzer loader
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
 */

#ifndef XCAM_ANALYZER_LOADER_H
#define XCAM_ANALYZER_LOADER_H

#include <base/xcam_common.h>
#include <base/xcam_3a_description.h>
#include "xcam_utils.h"
#include "x3a_analyzer.h"

namespace XCam {
class IspController;
class HybridAnalyzer;

class AnalyzerLoader
{
public:
    AnalyzerLoader (const char *lib_path);
    ~AnalyzerLoader ();

    SmartPtr<X3aAnalyzer> load_dynamic_analyzer (SmartPtr<AnalyzerLoader> &self);
    SmartPtr<X3aAnalyzer> load_hybrid_analyzer (SmartPtr<AnalyzerLoader> &self,
            SmartPtr<IspController> &isp,
            const char *cpf_path);

private:
    bool open_handle ();
    XCam3ADescription *get_symbol (const char *symbol);
    bool close_handle ();
    void convert_results (XCam3aResultHead *from, uint32_t num_of_from, X3aResultList &to);
    XCam3ADescription *load_analyzer (SmartPtr<AnalyzerLoader> &self);

    XCAM_DEAD_COPY(AnalyzerLoader);

private:
    char         *_path;
    void         *_handle;
};

};

#endif //XCAM_ANALYZER_LOADER_H
