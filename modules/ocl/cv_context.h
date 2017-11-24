/*
 * cv_context.h - used to init_opencv_ocl once
 *
 *  Copyright (c) 2016-2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Andrey Parfenov <a1994ndrey@gmail.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#ifndef XCAM_CV_CONTEXT_H
#define XCAM_CV_CONTEXT_H

#include <xcam_std.h>
#include <xcam_obj_debug.h>
#include <xcam_mutex.h>
#include <ocl/cl_context.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace XCam {

class CVContext
{
public:
    static SmartPtr<CVContext> instance ();

    SmartPtr<CLContext> get_cl_context () {
        return _context;
    }
    ~CVContext();
    bool enable_ocl (bool flag);
    bool is_ocl_enabled () const;

private:
    CVContext ();
    void init_opencv_ocl ();

    static Mutex                _init_mutex;
    static SmartPtr<CVContext>  _instance;

    SmartPtr<CLContext>         _context;

    XCAM_DEAD_COPY (CVContext);

};

}

#endif // XCAM_CV_CONTEXT_H
