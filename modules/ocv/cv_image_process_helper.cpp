/*
 * cv_image_process_helper.cpp - OpenCV image processing helpers functions
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

#include "cv_image_process_helper.h"

namespace XCam {


CVImageProcessHelper::CVImageProcessHelper ()
    : CVBaseClass ()
{

}

cv::Mat
CVImageProcessHelper::erosion (const cv::Mat &image, int erosion_size, int erosion_type)
{
    cv::Mat element = cv::getStructuringElement (erosion_type,
                      cv::Size (2 * erosion_size + 1, 2 * erosion_size + 1),
                      cv::Point (erosion_size, erosion_size));
    cv::Mat eroded;
    cv::erode (image, eroded, element);
    return eroded.clone ();
}

float
CVImageProcessHelper::get_snr (const cv::Mat &noisy, const cv::Mat &noiseless)
{
    cv::Mat temp_noisy, temp_noiseless;
    noisy.convertTo (temp_noisy, CV_32FC1);
    noiseless.convertTo (temp_noiseless, CV_32FC1);
    cv::Mat numerator, denominator;
    cv::pow (temp_noisy, 2, numerator);
    cv::pow (temp_noisy - temp_noiseless, 2, denominator);
    float res = cv::sum (numerator)[0] / cv::sum (denominator)[0];
    res = sqrt (res);
    return res;
}

void
CVImageProcessHelper::compute_dft (const cv::Mat &image, cv::Mat &result)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize (image.rows);
    int n = cv::getOptimalDFTSize (image.cols);
    cv::copyMakeBorder (image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float> (padded), cv::Mat::zeros (padded.size (), CV_32FC1)};
    cv::merge (planes, 2, result);
    cv::dft (result, result);
}

void
CVImageProcessHelper::compute_idft (cv::Mat *input, cv::Mat &result)
{
    cv::Mat fimg;
    cv::merge (input, 2, fimg);
    cv::idft (fimg, result, cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
}

void
CVImageProcessHelper::apply_constraints (cv::Mat &image, float threshold_min_value, float threshold_max_value, float min_value, float max_value)
{
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<float>(i, j) < threshold_min_value)
            {
                image.at<float>(i, j) = min_value;
            }
            if (image.at<float>(i, j) > threshold_max_value)
            {
                image.at<float>(i, j) = max_value;
            }
        }
    }
}

void
CVImageProcessHelper::normalize_weights (cv::Mat &weights)
{
    weights.convertTo (weights, CV_32FC1);
    float sum = cv::sum (weights)[0];
    weights /= sum;
}

}
