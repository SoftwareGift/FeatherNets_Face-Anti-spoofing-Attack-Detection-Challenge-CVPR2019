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
    : CVBaseClass()
{

}

void
CVEdgetaper::normalized_autocorrelation (const cv::Mat &psf, cv::Mat &auto_correlation_psf)
{
    cv::Mat correlation;
    cv::copyMakeBorder (psf, auto_correlation_psf, psf.cols - 1, 0, psf.rows - 1, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::filter2D (auto_correlation_psf, correlation, -1, psf, cv::Point(0, 0), 0, cv::BORDER_CONSTANT);
    cv::normalize (correlation, correlation, 0, 1.0f, cv::NORM_MINMAX);
    auto_correlation_psf = correlation.clone ();
}

void
CVEdgetaper::create_weights (cv::Mat &coefficients, const cv::Mat &psf)
{
    int psfr_last = psf.rows - 1;
    int psfc_last = psf.cols - 1;

    cv::Mat auto_correlation_psf;
    normalized_autocorrelation(psf, auto_correlation_psf);

    for (int i = 0; i < coefficients.rows; i++)
    {
        for (int j = 0; j < coefficients.cols; j++) {
            if (i < psfr_last)
            {
                if (j < psfc_last)
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i, j);
                else if (psfc_last <= j && j < (coefficients.cols - psfc_last))
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i, psfc_last);
                else
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i, j - (coefficients.cols - 2 * psfc_last) + 1);
            }
            else if (psfr_last <= i && i < (coefficients.rows - psfr_last))
            {
                if (j < psfc_last)
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(psfr_last, j);
                else if (psfc_last <= j && j < (coefficients.cols - psfc_last))
                    coefficients.at<float>(i, j) = 1.0f;
                else
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(psfr_last, j - (coefficients.cols - 2 * psfc_last) + 1);
            }
            else
            {
                if (j < psfc_last)
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i - (coefficients.rows - 2 * psfr_last) + 1, j);
                else if (psfc_last <= j && j < (coefficients.cols - psfc_last))
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i - (coefficients.rows - 2 * psfr_last) + 1, psfc_last);
                else
                    coefficients.at<float>(i, j) = auto_correlation_psf.at<float>(i - (coefficients.rows - 2 * psfr_last) + 1, j - (coefficients.cols - 2 * psfc_last) + 1);
            }
        }
    }
}

cv::Mat
CVEdgetaper::edgetaper (const cv::Mat &img, const cv::Mat &psf)
{
    cv::Mat blurred = cv::Mat::zeros (img.rows, img.cols, CV_32FC1);
    cv::filter2D (img, blurred, CV_32F, psf, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);

    cv::Mat coefficients = cv::Mat::zeros (img.rows, img.cols, CV_32FC1);
    create_weights (coefficients, psf);
    cv::Mat result = img.clone ();
    result.convertTo (result, CV_32FC1);
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (coefficients.at<float>(i, j) != 1.0f)
            {
                result.at<float>(i, j) = img.at<unsigned char>(i, j) * coefficients.at<float>(i, j) +
                                         blurred.at<float>(i, j) * (1.0f - coefficients.at<float>(i, j));
            }
        }
    }
    return result;
}

}
