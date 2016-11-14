/*
 * smart_analyzer_loader.cpp - smart analyzer loader
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

#include "smart_analyzer_loader.h"
#include "analyzer_loader.h"
#include "smart_analyzer.h"
#include "smart_analysis_handler.h"
#include <dirent.h>

namespace XCam {

#define MAX_PLUGIN_LIB_COUNT 10

SmartAnalyzerLoader::SmartAnalyzerLoader (const char *lib_path, const char *name, const char *symbol)
    : AnalyzerLoader (lib_path, symbol)
    , _name (NULL)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);
}

SmartAnalyzerLoader::~SmartAnalyzerLoader ()
{
    if (_name)
        xcam_free (_name);
}

SmartHandlerList

SmartAnalyzerLoader::load_smart_handlers (const char *dir_path)
{
    SmartHandlerList ret_handers;
    AnalyzerLoaderList loaders = create_analyzer_loader (dir_path);
    for (AnalyzerLoaderList::iterator i_loader = loaders.begin ();
            i_loader != loaders.end (); ++i_loader)
    {
        SmartPtr<SmartAnalysisHandler> handler = (*i_loader)->load_smart_handler(*i_loader);
        if (!handler.ptr ())
            continue;

        SmartHandlerList::iterator i_pos = ret_handers.begin ();
        for (; i_pos != ret_handers.end (); ++i_pos)
        {
            if (handler->get_priority() < (*i_pos)->get_priority ())
                break;
        }
        ret_handers.insert (i_pos, handler);
    }
    return ret_handers;
}

AnalyzerLoaderList
SmartAnalyzerLoader::create_analyzer_loader (const char *dir_path)
{
    XCAM_ASSERT (dir_path);

    char lib_path[512];
    DIR  *lib_dir = NULL;
    struct dirent *dirent_lib = NULL;
    SmartPtr<SmartAnalyzerLoader> loader;
    AnalyzerLoaderList loader_list;
    uint8_t count = 0;

    lib_dir = opendir (dir_path);
    if (lib_dir) {
        while ((count < MAX_PLUGIN_LIB_COUNT) && (dirent_lib = readdir (lib_dir)) != NULL) {
            if (dirent_lib->d_type != DT_LNK &&
                    dirent_lib->d_type != DT_REG)
                continue;
            snprintf (lib_path, sizeof(lib_path), "%s/%s", dir_path, dirent_lib->d_name);
            loader = new SmartAnalyzerLoader (lib_path, dirent_lib->d_name);
            if (loader.ptr ()) {
                loader_list.push_back (loader);
            }
        }
    }
    if (lib_dir)
        closedir (lib_dir);
    return loader_list;
}

SmartPtr<SmartAnalysisHandler>
SmartAnalyzerLoader::load_smart_handler (SmartPtr<SmartAnalyzerLoader> &self)
{
    XCAM_ASSERT (self.ptr () == this);

    SmartPtr<SmartAnalysisHandler> handler;
    XCamSmartAnalysisDescription *desc = (XCamSmartAnalysisDescription*)load_library (get_lib_path ());
    if (NULL == desc) {
        XCAM_LOG_WARNING ("load smart handler lib symbol failed");
        return NULL;
    }

    handler = new SmartAnalysisHandler (desc, self, (desc->name ? desc->name : _name));
    if (!handler.ptr ()) {
        XCAM_LOG_WARNING ("create smart handler failed");
        close_handle ();
        return NULL;
    }

    XCAM_LOG_INFO ("smart handler(%s) created from lib", XCAM_STR (handler->get_name()));
    return handler;
}

void *
SmartAnalyzerLoader::load_symbol (void* handle)
{
    XCamSmartAnalysisDescription *desc = NULL;

    desc = (XCamSmartAnalysisDescription *)AnalyzerLoader::get_symbol (handle);
    if (!desc) {
        XCAM_LOG_DEBUG ("get symbol failed from lib");
        return NULL;
    }
    if (desc->version < XCAM_VERSION) {
        XCAM_LOG_WARNING ("get symbol version is:0x%04x, but expect:0x%04x",
                          desc->version, XCAM_VERSION);
    }
    if (desc->size < sizeof (XCamSmartAnalysisDescription)) {
        XCAM_LOG_DEBUG ("get symbol failed, XCamSmartAnalysisDescription size is:%" PRIu32 ", but expect:%" PRIuS,
                        desc->size, sizeof (XCamSmartAnalysisDescription));
        return NULL;
    }

    if (!desc->create_context || !desc->destroy_context ||
            !desc->update_params || !desc->analyze ||
            !desc->free_results) {
        XCAM_LOG_DEBUG ("some functions in symbol not set from lib");
        return NULL;
    }
    return (void*)desc;
}

};
