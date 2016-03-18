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
#include <dlfcn.h>

namespace XCam {

AnalyzerLoader::AnalyzerLoader (const char *lib_path, const char *symbol)
    : _handle (NULL)
{
    XCAM_ASSERT (lib_path);
    _path = strndup (lib_path, XCAM_MAX_STR_SIZE);
    XCAM_ASSERT (symbol);
    _symbol = strndup (symbol, XCAM_MAX_STR_SIZE);
}

AnalyzerLoader::~AnalyzerLoader ()
{
    close_handle ();
    if (_path)
        xcam_free (_path);
    if (_symbol)
        xcam_free (_symbol);
}

void *
AnalyzerLoader::load_library (const char *lib_path)
{
    void *desc = NULL;

    void *handle = open_handle (lib_path);
    //XCAM_ASSERT (handle);
    if (!handle) {
        XCAM_LOG_WARNING ("open dynamic lib:%s failed", XCAM_STR (lib_path));
        return NULL;
    }
    desc = load_symbol (handle);
    if (!desc) {
        XCAM_LOG_WARNING ("get symbol(%s) from lib:%s failed", _symbol, XCAM_STR (lib_path));
        close_handle ();
        return NULL;
    }

    XCAM_LOG_DEBUG ("got symbols(%s) from lib(%s)", _symbol, XCAM_STR (lib_path));
    return desc;
}

void*
AnalyzerLoader::open_handle (const char *lib_path)
{
    void *handle = NULL;

    if (_handle != NULL)
        return _handle;

    handle = dlopen (lib_path, RTLD_LAZY);
    if (!handle) {
        XCAM_LOG_DEBUG (
            "open user-defined lib(%s) failed, reason:%s",
            XCAM_STR (lib_path), dlerror ());
        return NULL;
    }
    _handle = handle;
    return handle;
}

void *
AnalyzerLoader::get_symbol (void* handle)
{
    void *desc = NULL;

    XCAM_ASSERT (handle);
    XCAM_ASSERT (_symbol);
    desc = (void *)dlsym (handle, _symbol);
    if (!desc) {
        XCAM_LOG_DEBUG ("get symbol(%s) failed from lib(%s), reason:%s", _symbol, XCAM_STR (_path), dlerror ());
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
