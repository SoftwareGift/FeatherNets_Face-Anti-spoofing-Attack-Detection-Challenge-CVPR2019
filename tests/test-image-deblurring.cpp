/*
 * test-image-deblurring.cpp - test image deblurring
 *
 *  Copyright (c) 2017 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
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

#include "test_common.h"
#include "test_inline.h"

#include <unistd.h>
#include <getopt.h>
#include <image_file_handle.h>
#include "ocv/cv_image_sharp.h"
#include "ocv/cv_wiener_filter.h"
#include "ocv/cv_image_deblurring.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace XCam;

static void
usage (const char* arg0)
{
    printf ("Usage: %s --input file --output file\n"
            "\t--input,    input image(RGB)\n"
            "\t--output,   output image(RGB) PREFIX\n"
            "\t--blind,    optional, blind or non-blind deblurring, default true; select from [true/false]\n"
            "\t--save,     optional, save file or not, default true; select from [true/false]\n"
            "\t--help,     usage\n",
            arg0);
}

static void
blind_deblurring (cv::Mat &input_image, cv::Mat &output_image)
{
    SmartPtr<CVImageDeblurring> image_deblurring = new CVImageDeblurring ();
    cv::Mat kernel;
    image_deblurring->blind_deblurring (input_image, output_image, kernel, -1, -1, false);
}

static void
non_blind_deblurring (cv::Mat &input_image, cv::Mat &output_image)
{
    SmartPtr<CVWienerFilter> wiener_filter = new CVWienerFilter ();
    cv::cvtColor (input_image, input_image, cv::COLOR_BGR2GRAY);
    // use simple motion blur kernel
    int kernel_size = 13;
    cv::Mat kernel = cv::Mat::zeros (kernel_size, kernel_size, CV_32FC1);
    for (int i = 0; i < kernel_size; i++)
    {
        kernel.at<float> ((kernel_size - 1) / 2, i) = 1.0;
    }
    kernel /= kernel_size;
    //flip kernel to perform convolution
    cv::Mat conv_kernel;
    cv::flip (kernel, conv_kernel, -1);
    cv::Mat blurred;
    cv::filter2D (input_image, blurred, CV_32FC1, conv_kernel, cv::Point(-1, -1), 0, cv::BORDER_CONSTANT);
    // restore the image
    cv::Mat median_blurred;
    medianBlur (blurred, median_blurred, 3);
    SmartPtr<CVImageProcessHelper> helpers = new CVImageProcessHelper ();
    float noise_power = 1.0f / helpers->get_snr (blurred, median_blurred);
    wiener_filter->wiener_filter (blurred, kernel, output_image, noise_power);
}

int main (int argc, char *argv[])
{
    const char *file_in_name = NULL;
    const char *file_out_name = NULL;

    bool need_save_output = true;
    bool blind = true;

    const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"blind", required_argument, NULL, 'b'},
        {"save", required_argument, NULL, 's'},
        {"help", no_argument, NULL, 'H'},
        {0, 0, 0, 0},
    };

    int opt = -1;
    while ((opt = getopt_long (argc, argv, "", long_opts, NULL)) != -1)
    {
        switch (opt) {
        case 'i':
            file_in_name = optarg;
            break;
        case 'o':
            file_out_name = optarg;
            break;
        case 'b':
            blind = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 's':
            need_save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'H':
            usage (argv[0]);
            return -1;
        default:
            printf ("getopt_long return unknown value:%c\n", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2)
    {
        printf ("unknown option %s\n", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    if (!file_in_name || !file_out_name)
    {
        XCAM_LOG_ERROR ("input/output path is NULL");
        return -1;
    }

    printf ("Description-----------\n");
    printf ("input image file:%s\n", file_in_name);
    printf ("output file :%s\n", file_out_name);
    printf ("blind deblurring:%s\n", blind ? "true" : "false");
    printf ("need save file:%s\n", need_save_output ? "true" : "false");
    printf ("----------------------\n");

    SmartPtr<CVImageSharp> sharp = new CVImageSharp ();
    cv::Mat input_image = cv::imread (file_in_name, cv::IMREAD_COLOR);
    cv::Mat output_image;
    if (input_image.empty ())
    {
        XCAM_LOG_ERROR ("input file read error");
        return -1;
    }
    if (blind)
    {
        blind_deblurring (input_image, output_image);
    }
    else
    {
        non_blind_deblurring (input_image, output_image);
    }
    float input_sharp = sharp->measure_sharp (input_image);
    float output_sharp = sharp->measure_sharp (output_image);
    if (need_save_output)
    {
        cv::imwrite (file_out_name, output_image);
    }
    XCAM_ASSERT (output_sharp > input_sharp);

    return 0;
}

