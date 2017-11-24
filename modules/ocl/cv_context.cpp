/*
 * cv_context.cpp - used to init_opencv_ocl once
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

#include "cv_context.h"
#include "cl_device.h"
#include "cl_memory.h"

namespace XCam {

Mutex CVContext::_init_mutex;
SmartPtr<CVContext> CVContext::_instance;


SmartPtr<CVContext>
CVContext::instance ()
{
    SmartLock locker (_init_mutex);
    if (_instance.ptr())
        return _instance;

    _instance = new CVContext ();
    _instance->init_opencv_ocl ();
    return _instance;
}

void
CVContext::init_opencv_ocl ()
{
    _context = CLDevice::instance()->get_context();
    cl_platform_id platform_id = CLDevice::instance()->get_platform_id ();
    char *platform_name = CLDevice::instance()->get_platform_name ();
    cl_device_id device_id = CLDevice::instance()->get_device_id ();
    cl_context _context_id = _context->get_context_id ();
    cv::ocl::attachContext (platform_name, platform_id, _context_id, device_id);
    cv::ocl::setUseOpenCL (cv::ocl::haveOpenCL());
    XCAM_LOG_DEBUG("Use OpenCL is:  %s", cv::ocl::haveOpenCL() ? "true" : "false");
}

bool
CVContext::enable_ocl (bool flag)
{
    if (flag && !cv::ocl::haveOpenCL()) {
        return false;
    }
    cv::ocl::setUseOpenCL (flag);
    return true;
}

bool
CVContext::is_ocl_enabled () const
{
    return cv::ocl::useOpenCL ();
}

CVContext::CVContext ()
{

}

CVContext::~CVContext () {

}

}
