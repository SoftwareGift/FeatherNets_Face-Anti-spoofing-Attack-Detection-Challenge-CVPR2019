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

namespace XCam {

class AnalyzerLoader
{
public:
    AnalyzerLoader (const char *lib_path, const char *symbol);
    virtual ~AnalyzerLoader ();

protected:
    void *load_library (const char *lib_path);
    void *get_symbol (void* handle);
    virtual void *load_symbol (void* handle) = 0;
    bool close_handle ();
    const char * get_lib_path () const {
        return _path;
    }

private:
    void *open_handle (const char *lib_path);

    XCAM_DEAD_COPY(AnalyzerLoader);

private:
    void *_handle;
    char *_symbol;
    char *_path;
};

};

#endif //XCAM_ANALYZER_LOADER_H
