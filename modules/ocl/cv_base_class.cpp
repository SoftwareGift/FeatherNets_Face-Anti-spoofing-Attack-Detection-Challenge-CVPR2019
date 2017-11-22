/*
 * cv_base_class.cpp - base class for all OpenCV related features
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

#include "cv_base_class.h"

namespace XCam {

CVBaseClass::CVBaseClass ()
{
    _cv_context = CVContext::instance ();
    XCAM_ASSERT (_cv_context.ptr ());
    _use_ocl = _cv_context->is_ocl_enabled ();
}

bool
CVBaseClass::set_ocl (bool use_ocl)
{
    if (use_ocl && !_cv_context->is_ocl_enabled ()) {
        return false;
    }
    _use_ocl = use_ocl;
    return true;
}

bool
CVBaseClass::convert_to_mat (SmartPtr<VideoBuffer> buffer, cv::Mat &image)
{

    VideoBufferInfo info = buffer->get_video_info ();
    XCAM_FAIL_RETURN (WARNING, info.format == V4L2_PIX_FMT_NV12, false, "convert_to_mat only support NV12 format");

    uint8_t *ptr = buffer->map ();
    XCAM_FAIL_RETURN (WARNING, ptr, false, "convert_to_mat buffer map failed");

    cv::Mat mat = cv::Mat (info.aligned_height * 3 / 2, info.width, CV_8UC1, ptr, info.strides[0]);
    cv::cvtColor (mat, image, cv::COLOR_YUV2BGR_NV12);
    //buffer->unmap ();

    return true;
}

bool
convert_to_mat (SmartPtr<VideoBuffer> buffer, cv::Mat &image)
{
    CVBaseClass cv_obj;
    return cv_obj.convert_to_mat (buffer, image);
}

}
