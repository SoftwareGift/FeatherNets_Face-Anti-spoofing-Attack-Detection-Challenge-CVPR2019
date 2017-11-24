/*
 * x3a_analyzer_manager.h - analyzer manager
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

#ifndef XCAM_3A_ANALYZER_MANAGER_H
#define XCAM_3A_ANALYZER_MANAGER_H

#include <xcam_std.h>
#include <x3a_analyzer.h>
#include <dynamic_analyzer_loader.h>

namespace XCam {

class DynamicAnalyzerLoader;

class X3aAnalyzerManager
{
protected:
    explicit X3aAnalyzerManager ();
public:
    virtual ~X3aAnalyzerManager ();

    static SmartPtr<X3aAnalyzerManager> instance();

    virtual SmartPtr<X3aAnalyzer> create_analyzer();

private:
    SmartPtr<X3aAnalyzer> find_analyzer ();
    SmartPtr<X3aAnalyzer> load_analyzer_from_binary (const char *path);

private:
    XCAM_DEAD_COPY (X3aAnalyzerManager);

private:
    static SmartPtr<X3aAnalyzerManager> _instance;
    static Mutex                        _mutex;

    SmartPtr<DynamicAnalyzerLoader>     _loader;
};
};
#endif //XCAM_3A_ANALYZER_MANAGER_H
