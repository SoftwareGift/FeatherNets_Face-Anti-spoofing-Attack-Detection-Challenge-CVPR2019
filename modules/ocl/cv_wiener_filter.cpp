/*
 * cv_image_deblurring.cpp - iterative blind deblurring
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

#include "cv_wiener_filter.h"

namespace XCam {


CVWienerFilter::CVWienerFilter ()
    : CVBaseClass ()
{
    _helpers = new CVImageProcessHelper ();
}

void
CVWienerFilter::wiener_filter (const cv::Mat &blurred_image, const cv::Mat &known, cv::Mat &unknown, float noise_power)
{
    int image_w = blurred_image.size ().width;
    int image_h = blurred_image.size ().height;
    cv::Mat y_ft;
    _helpers->compute_dft (blurred_image, y_ft);

    cv::Mat padded = cv::Mat::zeros (image_h, image_w, CV_32FC1);
    int padx = padded.cols - known.cols;
    int pady = padded.rows - known.rows;
    cv::copyMakeBorder (known, padded, 0, pady, 0, padx, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat padded_ft;
    _helpers->compute_dft (padded, padded_ft);

    cv::Mat temp_unknown;
    cv::Mat unknown_ft[2];
    unknown_ft[0] = cv::Mat::zeros (image_h, image_w, CV_32FC1);
    unknown_ft[1] = cv::Mat::zeros (image_h, image_w, CV_32FC1);

    cv::Mat denominator;
    cv::Mat denominator_splitted[] = {cv::Mat::zeros (blurred_image.size (), CV_32FC1), cv::Mat::zeros (blurred_image.size (), CV_32FC1)};
    cv::mulSpectrums (padded_ft, padded_ft, denominator, 0, true);
    cv::split (denominator, denominator_splitted);
    denominator_splitted[0] = denominator_splitted[0] (cv::Rect (0, 0, blurred_image.cols, blurred_image.rows));
    denominator_splitted[0] += cv::Scalar (noise_power);

    cv::Mat numerator;
    cv::Mat numerator_splitted[] = {cv::Mat::zeros (blurred_image.size (), CV_32FC1), cv::Mat::zeros (blurred_image.size (), CV_32FC1)};
    cv::mulSpectrums (y_ft, padded_ft, numerator, 0, true);
    cv::split (numerator, numerator_splitted);
    numerator_splitted[0] = numerator_splitted[0] (cv::Rect (0, 0, blurred_image.cols, blurred_image.rows));
    numerator_splitted[1] = numerator_splitted[1] (cv::Rect (0, 0, blurred_image.cols, blurred_image.rows));
    cv::divide (numerator_splitted[0], denominator_splitted[0], unknown_ft[0]);
    cv::divide (numerator_splitted[1], denominator_splitted[0], unknown_ft[1]);
    _helpers->compute_idft (unknown_ft, temp_unknown);
    unknown = temp_unknown.clone();
}

}
