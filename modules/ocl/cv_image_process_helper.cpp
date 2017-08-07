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
    : CVBaseClass()
{

}

cv::Mat
CVImageProcessHelper::erosion (const cv::Mat &image, int erosion_size, int erosion_type)
{
    cv::Mat element = cv::getStructuringElement (erosion_type,
                      cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1 ),
                      cv::Point(erosion_size, erosion_size));
    cv::Mat eroded;
    cv::erode (image, eroded, element);
    return eroded.clone();
}

float
CVImageProcessHelper::get_snr (const cv::Mat &noisy, const cv::Mat &noiseless)
{
    cv::Mat temp_noisy, temp_noiseless;
    noisy.convertTo (temp_noisy, CV_8UC1);
    noiseless.convertTo (temp_noiseless, CV_8UC1);
    float numerator = 0;
    float denominator = 0;
    float res = 0;
    for (int i = 0; i < temp_noisy.rows; i++)
    {
        for (int j = 0; j < temp_noisy.cols; j++)
        {
            denominator += ((temp_noisy.at<unsigned char>(i, j) - temp_noiseless.at<unsigned char>(i, j))
                            * (temp_noisy.at<unsigned char>(i, j) - temp_noiseless.at<unsigned char>(i, j)));
            numerator += (temp_noisy.at<unsigned char>(i, j) * temp_noisy.at<unsigned char>(i, j));
        }
    }
    res = sqrt (numerator / denominator);
    return res;
}

cv::Mat
CVImageProcessHelper::get_auto_correlation (const cv::Mat &image)
{
    cv::Mat dst;
    cv::Laplacian (image, dst, -1, 3, 1, 0, cv::BORDER_CONSTANT);
    dst.convertTo (dst, CV_32FC1);
    cv::Mat correlation;
    cv::filter2D (dst, correlation, -1, dst, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    return correlation.clone ();
}

void
CVImageProcessHelper::compute_dft (const cv::Mat &image, cv::Mat *result)
{
    cv::Mat padded;
    int m = cv::getOptimalDFTSize (image.rows);
    int n = cv::getOptimalDFTSize (image.cols);
    cv::copyMakeBorder (image, padded, 0, m - image.rows, 0, n - image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32FC1)};
    cv::Mat fimg;
    cv::merge (planes, 2, fimg);
    cv::dft (fimg, fimg);
    cv::split (fimg, planes);
    planes[0] = planes[0] (cv::Rect(0, 0, image.cols, image.rows));
    planes[1] = planes[1] (cv::Rect(0, 0, image.cols, image.rows));
    result[0] = planes[0].clone ();
    result[1] = planes[1].clone ();
}

void
CVImageProcessHelper::compute_idft (cv::Mat *input, cv::Mat &result)
{
    cv::Mat fimg;
    cv::merge (input, 2, fimg);
    cv::Mat inverse;
    cv::idft (fimg, inverse, cv::DFT_REAL_OUTPUT + cv::DFT_SCALE);
    result = inverse.clone ();
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

// weights will be symmetric and sum(weights elements) == 1
void
CVImageProcessHelper::normalize_weights (cv::Mat &weights)
{
    weights.convertTo (weights, CV_32FC1);
    float sum = 0;
    for (int i = 0; i < weights.rows; i++)
    {
        for (int j = 0; j <= i; j++)
        {

            weights.at<float>(i, j) = (weights.at<float>(i, j) + weights.at<float>(j, i)) / 2;
            weights.at<float>(j, i) = weights.at<float>(i, j);
            if (j == i)
                sum += weights.at<float>(i, j);
            else
                sum += (2 * weights.at<float>(i, j));
        }
    }
    weights /= sum;
}

}
