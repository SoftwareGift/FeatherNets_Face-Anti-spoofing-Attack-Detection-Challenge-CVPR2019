/*
 * hybrid_analyzer_loader.h - hybrid analyzer loader
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

#ifndef XCAM_HYBRID_ANALYZER_LOADER_H
#define XCAM_HYBRID_ANALYZER_LOADER_H

#include <xcam_std.h>
#include <base/xcam_3a_description.h>
#include "dynamic_analyzer_loader.h"
#include "isp_controller.h"

namespace XCam {
class IspController;
class X3aAnalyzer;

class HybridAnalyzerLoader
    : public AnalyzerLoader
{
public:
    HybridAnalyzerLoader (const char *lib_path, const char *symbol = XCAM_3A_LIB_DESCRIPTION);
    virtual ~HybridAnalyzerLoader ();

    virtual bool set_cpf_path (const char *cpf_path);
    virtual bool set_isp_controller (SmartPtr<IspController> &isp);
    virtual SmartPtr<X3aAnalyzer> load_analyzer (SmartPtr<AnalyzerLoader> &self);

protected:
    virtual void *load_symbol (void* handle);

private:
    XCAM_DEAD_COPY(HybridAnalyzerLoader);

private:
    const char                    *_cpf_path;
    SmartPtr<IspController>       _isp;
};

};

#endif // XCAM_HYBRID_ANALYZER_LOADER_H