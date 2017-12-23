/*
 * cv_edgetaper.cpp - used in deblurring to remove ringing artifacts
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

#include "cv_edgetaper.h"

namespace XCam {


CVEdgetaper::CVEdgetaper ()
    : CVBaseClass ()
{

}

void
CVEdgetaper::create_weights (const cv::Mat &image, const cv::Mat &psf, cv::Mat &coefficients)
{
    cv::Mat rows_proj, cols_proj;
    cv::Mat rows_proj_border, cols_proj_border;
    cv::Mat rows_cor, cols_cor;
    // get psf rows and cols projections
    cv::reduce (psf, rows_proj, 1, CV_REDUCE_SUM, -1);
    cv::reduce (psf, cols_proj, 0, CV_REDUCE_SUM, -1);
    // calculate correlation for psf projections
    cv::copyMakeBorder (rows_proj, rows_proj_border, (psf.rows - 1) / 2, (psf.rows - 1) / 2, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all (0));
    cv::copyMakeBorder (cols_proj, cols_proj_border, 0, 0,  (psf.cols - 1) / 2, (psf.cols - 1) / 2, cv::BORDER_CONSTANT, cv::Scalar::all (0));
    cv::matchTemplate (rows_proj_border, rows_proj, rows_cor, CV_TM_CCORR);
    cv::matchTemplate (cols_proj_border, cols_proj, cols_cor, CV_TM_CCORR);
    // make it symmetric on both sides
    cv::Mat rows_add = cv::Mat_<float>(1, 1) << rows_proj.at<float> (0, 0);
    cv::Mat cols_add = cv::Mat_<float>(1, 1) << cols_proj.at<float> (0, 0);
    cv::vconcat (rows_cor, rows_add, rows_cor);
    cv::hconcat (cols_cor, cols_add, cols_cor);
    double min, max;
    cv::minMaxLoc (rows_cor, &min, &max);
    rows_cor /= max;
    cv::minMaxLoc (cols_cor, &min, &max);
    cols_cor /= max;
    // get matrix from projections
    cv::Mat alpha = (cv::Scalar (1) - rows_proj) * (cv::Scalar (1) - cols_proj);
    // expand it to the image size
    int nc = image.cols / psf.cols + 1;
    int nr = image.rows / psf.rows + 1;
    cv::Mat expanded;
    cv::repeat (alpha, nr, nc, expanded);
    cv::Mat weights = expanded (cv::Rect (expanded.cols / 2 - image.cols / 2, expanded.rows / 2 - image.rows / 2, image.cols, image.rows));
    coefficients = weights.clone ();
}

void
CVEdgetaper::edgetaper (const cv::Mat &img, const cv::Mat &psf, cv::Mat &output)
{
    cv::Mat blurred = cv::Mat::zeros (img.rows, img.cols, CV_32FC1);
    // flip PSF to perform convolution
    cv::Mat psf_flipped;
    cv::flip (psf, psf_flipped, -1);
    cv::filter2D (img, blurred, CV_32FC1, psf_flipped, cv::Point (-1, -1), 0, cv::BORDER_CONSTANT);
    cv::Mat coefficients;
    create_weights (img, psf, coefficients);
    cv::Mat result;
    img.convertTo (result, CV_32FC1);
    result = result.mul (coefficients) + blurred.mul (cv::Scalar (1.0f) - coefficients);
    output = result.clone ();
}

}
