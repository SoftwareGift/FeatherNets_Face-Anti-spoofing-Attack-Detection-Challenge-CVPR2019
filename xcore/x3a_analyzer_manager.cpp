/*
 * x3a_analyzer_manager.cpp - analyzer manager
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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

#include "x3a_analyzer_manager.h"
#include "x3a_analyzer_simple.h"
#include <sys/types.h>
#include <dirent.h>

namespace XCam {

#define XCAM_DEFAULT_3A_LIB_DIR "/usr/lib/xcam/plugins/3a"

SmartPtr<X3aAnalyzerManager> X3aAnalyzerManager::_instance(NULL);
Mutex X3aAnalyzerManager::_mutex;

SmartPtr<X3aAnalyzerManager>
X3aAnalyzerManager::instance()
{
    SmartLock lock(_mutex);
    if (_instance.ptr())
        return _instance;
    _instance = new X3aAnalyzerManager;
    return _instance;
}

X3aAnalyzerManager::X3aAnalyzerManager ()
{
    XCAM_LOG_DEBUG ("X3aAnalyzerManager construction");
}
X3aAnalyzerManager::~X3aAnalyzerManager ()
{
    XCAM_LOG_DEBUG ("X3aAnalyzerManager destruction");
}

SmartPtr<X3aAnalyzer>
X3aAnalyzerManager::create_analyzer()
{
    SmartPtr<X3aAnalyzer> analyzer = find_analyzer();
    if (!analyzer.ptr())
        analyzer = new X3aAnalyzerSimple;
    return analyzer;
}

SmartPtr<X3aAnalyzer>
X3aAnalyzerManager::find_analyzer ()
{
    char lib_path[512];
    const char *dir_path = NULL;
    DIR  *dir_3a = NULL;
    struct dirent *dirent_3a = NULL;
    SmartPtr<X3aAnalyzer> analyzer;

    dir_path = getenv ("XCAM_3A_LIB");
    if (!dir_path) {
        dir_path = XCAM_DEFAULT_3A_LIB_DIR;
        XCAM_LOG_INFO ("doesn't find environment=>XCAM_3A_LIB, change to default dir:%s", dir_path);
    }
    dir_3a = opendir (dir_path);
    if (dir_3a) {
        while ((dirent_3a = readdir (dir_3a)) != NULL) {
            if (dirent_3a->d_type != DT_LNK &&
                    dirent_3a->d_type != DT_REG)
                continue;
            snprintf (lib_path, sizeof(lib_path), "%s/%s", dir_path, dirent_3a->d_name);
            analyzer = load_analyzer_from_binary (lib_path);
            if (analyzer.ptr())
                break;
        }
    }
    if (dir_3a)
        closedir (dir_3a);
    return analyzer;
}

SmartPtr<X3aAnalyzer>
X3aAnalyzerManager::load_analyzer_from_binary (const char *path)
{
    SmartPtr<X3aAnalyzer> analyzer;

    XCAM_ASSERT (path);

    _loader.release ();
    _loader = new DynamicAnalyzerLoader (path);

    SmartPtr<AnalyzerLoader> loader = _loader.dynamic_cast_ptr<AnalyzerLoader> ();
    analyzer = _loader->load_analyzer (loader);

    if (analyzer.ptr ())
        return analyzer;

    XCAM_LOG_WARNING ("load 3A analyzer failed from: %s", path);
    return NULL;
}

};

