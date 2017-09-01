/*
 * cv_sharp.h - sharp image
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

#ifndef XCAM_CV_IMAGE_SHARP_H
#define XCAM_CV_IMAGE_SHARP_H

#include "xcam_utils.h"
#include <base/xcam_common.h>
#include <base/xcam_buffer.h>
#include <dma_video_buffer.h>
#include <smartptr.h>
#include "xcam_obj_debug.h"
#include "image_file_handle.h"
#include "cv_base_class.h"
#include "cv_image_process_helper.h"

#include <ocl/cl_context.h>
#include <ocl/cl_device.h>
#include <ocl/cl_memory.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

namespace XCam {

class CVImageSharp : public CVBaseClass
{

public:
    explicit CVImageSharp ();

    float measure_sharp (const cv::Mat &image);
    cv::Mat sharp_image_gray (const cv::Mat &image, float sigmar);

    XCAM_DEAD_COPY (CVImageSharp);
};

}

#endif // XCAM_CV_IMAGE_SHARP_H
