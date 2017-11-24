/*
 * cv_base_class.h - base class for all OpenCV related features
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

#ifndef XCAM_CV_BASE_CLASS_H
#define XCAM_CV_BASE_CLASS_H

#include <xcam_std.h>
#include <video_buffer.h>
#include <ocl/cv_context.h>

namespace XCam {

class CVBaseClass
{
public:
    explicit CVBaseClass ();
    bool set_ocl (bool use_ocl);
    bool is_ocl_path () {
        return _use_ocl;
    }
    bool convert_to_mat (SmartPtr<VideoBuffer> buffer, cv::Mat &image);

protected:
    XCAM_DEAD_COPY (CVBaseClass);
    SmartPtr<CVContext>  _cv_context;
    bool                 _use_ocl;
};

extern bool
convert_to_mat (SmartPtr<VideoBuffer> buffer, cv::Mat &image);

}

#endif // XCAM_CV_BASE_CLASS_H
