/*
 * hybrid_analyzer_loader.cpp - hybrid analyzer loader
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

#include "hybrid_analyzer_loader.h"
#include "handler_interface.h"
#include "hybrid_analyzer.h"
#include <dlfcn.h>

namespace XCam {

HybridAnalyzerLoader::HybridAnalyzerLoader (const char *lib_path, const char *symbol)
    : AnalyzerLoader (lib_path, symbol)
    , _cpf_path (NULL)
{
}

HybridAnalyzerLoader::~HybridAnalyzerLoader ()
{
    _isp.release ();
}

bool
HybridAnalyzerLoader::set_cpf_path (const char *cpf_path)
{
    XCAM_ASSERT (cpf_path && !_cpf_path);
    _cpf_path = cpf_path;
    return true;
}

bool
HybridAnalyzerLoader::set_isp_controller (SmartPtr<IspController> &isp)
{
    XCAM_ASSERT (isp.ptr() && !_isp.ptr());
    _isp = isp;
    return true;
}

SmartPtr<X3aAnalyzer>
HybridAnalyzerLoader::load_analyzer (SmartPtr<AnalyzerLoader> &self)
{
    XCAM_ASSERT (self.ptr () == this);

    SmartPtr<X3aAnalyzer> analyzer;

#if HAVE_IA_AIQ
    XCam3ADescription *desc = (XCam3ADescription*)load_library (get_lib_path ());
    analyzer = new HybridAnalyzer (desc, self, _isp, _cpf_path);
#endif

    if (!analyzer.ptr ()) {
        XCAM_LOG_WARNING ("create HybridAnalyzer from lib failed");
        close_handle ();
        return NULL;
    }

    XCAM_LOG_INFO ("analyzer(%s) created from 3a lib", XCAM_STR (analyzer->get_name()));
    return analyzer;
}

void *
HybridAnalyzerLoader::load_symbol (void* handle)
{
    XCam3ADescription *desc = NULL;

    desc = (XCam3ADescription *)AnalyzerLoader::get_symbol (handle);
    if (!desc) {
        XCAM_LOG_DEBUG ("get symbol failed from lib");
        return NULL;
    }
    if (desc->version < xcam_version ()) {
        XCAM_LOG_DEBUG ("get symbolfailed. version is:0x%04x, but expect:0x%04x",
                        desc->version, xcam_version ());
        return NULL;
    }
    if (desc->size < sizeof (XCam3ADescription)) {
        XCAM_LOG_DEBUG ("get symbol failed, XCam3ADescription size is:%" PRIu32 ", but expect:%" PRIuS,
                        desc->size, sizeof (XCam3ADescription));
        return NULL;
    }

    if (!desc->create_context || !desc->destroy_context ||
            !desc->configure_3a || !desc->set_3a_stats ||
            !desc->analyze_awb || !desc->analyze_ae ||
            !desc->analyze_af || !desc->combine_analyze_results ||
            !desc->free_results) {
        XCAM_LOG_DEBUG ("some functions in symbol not set from lib");
        return NULL;
    }
    return (void*)desc;
}

};
