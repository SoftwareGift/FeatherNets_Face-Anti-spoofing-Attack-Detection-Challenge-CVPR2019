/*
 * cv_feature_match.h - optical flow feature match
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_CV_FEATURE_MATCH_H
#define XCAM_CV_FEATURE_MATCH_H

#include <base/xcam_common.h>
#include <base/xcam_buffer.h>
#include <dma_video_buffer.h>
#include <smartptr.h>

#include <cl_context.h>
#include <cl_device.h>
#include <cl_memory.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace XCam;

void init_opencv_ocl (SmartPtr<CLContext> context);

bool convert_to_mat (SmartPtr<CLContext> context, SmartPtr<DrmBoBuffer> buffer, cv::Mat &mat);

void optical_flow_feature_match (
    SmartPtr<CLContext> context, int output_width,
    SmartPtr<DrmBoBuffer> buf0, SmartPtr<DrmBoBuffer> buf1,
    cv::Rect &image0_crop_left, cv::Rect &image0_crop_right,
    cv::Rect &image1_crop_left, cv::Rect &image1_crop_right);

#endif // XCAM_CV_FEATURE_MATCH_H
