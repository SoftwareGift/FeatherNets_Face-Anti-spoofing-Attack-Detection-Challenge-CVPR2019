/*
 * analyzer_loader.cpp - analyzer loader
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

#include "analyzer_loader.h"
#include "dynamic_analyzer.h"
#include "handler_interface.h"
#include "hybrid_analyzer.h"
#include "isp_controller.h"
#include "x3a_result_factory.h"
#include "x3a_statistics_queue.h"
#include <dlfcn.h>

namespace XCam {

AnalyzerLoader::AnalyzerLoader (const char *lib_path)
    : _path (NULL)
    , _handle (NULL)
{
    XCAM_ASSERT (lib_path);
    _path = strdup (lib_path);
}

AnalyzerLoader::~AnalyzerLoader ()
{
    close_handle ();
    if (_path)
        xcam_free (_path);
}

XCam3ADescription *
AnalyzerLoader::load_analyzer (SmartPtr<AnalyzerLoader> &self)
{
    XCam3ADescription *desc = NULL;
    const char *symbol = XCAM_3A_LIB_DESCRIPTION;

    XCAM_ASSERT (self.ptr () == this);

    if (!open_handle ()) {
        XCAM_LOG_WARNING ("open dynamic lib:%s failed", XCAM_STR (_path));
        return NULL;
    }

    desc = get_symbol (symbol);
    if (!desc) {
        XCAM_LOG_WARNING ("get symbol(%s) from lib:%s failed", symbol, XCAM_STR (_path));
        close_handle ();
        return NULL;
    }

    XCAM_LOG_DEBUG ("got symbols(%s) from lib(%s)", symbol, XCAM_STR (_path));
    return desc;
}

SmartPtr<X3aAnalyzer>
AnalyzerLoader::load_dynamic_analyzer (SmartPtr<AnalyzerLoader> &self)
{
    SmartPtr<X3aAnalyzer> analyzer;
    XCam3ADescription *desc = load_analyzer (self);

    analyzer = new DynamicAnalyzer (desc, self);
    if (!analyzer.ptr ()) {
        XCAM_LOG_WARNING ("create analyzer(%s) from lib:%s failed", XCAM_STR (analyzer->get_name()), XCAM_STR (_path));
        close_handle ();
        return NULL;
    }

    XCAM_LOG_INFO ("analyzer(%s) created from 3a lib(%s)", XCAM_STR (analyzer->get_name()), XCAM_STR (_path));
    return analyzer;
}

SmartPtr<X3aAnalyzer>
AnalyzerLoader::load_hybrid_analyzer (SmartPtr<AnalyzerLoader> &self,
                                      SmartPtr<IspController> &isp,
                                      const char *cpf_path)
{
    SmartPtr<X3aAnalyzer> analyzer;
    XCam3ADescription *desc = load_analyzer (self);

    analyzer = new HybridAnalyzer (desc, self, isp, cpf_path);
    if (!analyzer.ptr ()) {
        XCAM_LOG_WARNING ("create analyzer(%s) from lib:%s failed", XCAM_STR (analyzer->get_name()), XCAM_STR (_path));
        close_handle ();
        return NULL;
    }

    XCAM_LOG_INFO ("analyzer(%s) created from 3a lib(%s)", XCAM_STR (analyzer->get_name()), XCAM_STR (_path));
    return analyzer;
}

bool
AnalyzerLoader::open_handle ()
{
    void *handle = NULL;

    if (_handle != NULL)
        return true;

    handle = dlopen (_path, RTLD_LAZY);
    if (!handle) {
        XCAM_LOG_DEBUG (
            "open user-defined 3a lib(%s) failed, reason:%s",
            XCAM_STR (_path), dlerror ());
        return false;
    }
    _handle = handle;
    return true;
}

XCam3ADescription *
AnalyzerLoader::get_symbol (const char *symbol)
{
    XCam3ADescription *desc = NULL;

    XCAM_ASSERT (_handle);
    XCAM_ASSERT (symbol);
    desc = (XCam3ADescription *)dlsym (_handle, symbol);
    if (!desc) {
        XCAM_LOG_DEBUG ("get symbol(%s) failed from lib(%s), reason:%s", symbol, XCAM_STR (_path), dlerror ());
        return NULL;
    }
    if (desc->version < XCAM_VERSION) {
        XCAM_LOG_DEBUG ("get symbol(%s) failed. version is:0x%04x, but expect:0x%04x",
                        symbol, desc->version, XCAM_VERSION);
        return NULL;
    }
    if (desc->size < sizeof (XCam3ADescription)) {
        XCAM_LOG_DEBUG ("get symbol(%s) failed, XCam3ADescription size is:%d, but expect:%d",
                        symbol, desc->size, sizeof (XCam3ADescription));
        return NULL;
    }

    if (!desc->create_context || !desc->destroy_context ||
            !desc->configure_3a || !desc->set_3a_stats ||
            !desc->analyze_awb || !desc->analyze_ae ||
            !desc->analyze_af || !desc->combine_analyze_results ||
            !desc->free_results) {
        XCAM_LOG_DEBUG ("some functions in symbol(%s) not set from lib(%s)", symbol, XCAM_STR (_path));
        return NULL;
    }
    return desc;
}

bool
AnalyzerLoader::close_handle ()
{
    if (!_handle)
        return true;
    dlclose (_handle);
    _handle = NULL;
    return true;
}

};
