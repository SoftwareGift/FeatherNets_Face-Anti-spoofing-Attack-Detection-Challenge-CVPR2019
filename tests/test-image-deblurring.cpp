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
#include "ocl/cl_device.h"
#include "ocl/cl_context.h"
#include "ocl/cl_blender.h"
#include "image_file_handle.h"
#include "ocl/cv_image_deblurring.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>

using namespace XCam;

static void
usage(const char* arg0)
{
    printf ("Usage: %s --input file --output file\n"
            "\t--input,    input image(RGB)\n"
            "\t--output,   output image(RGB) PREFIX\n"
            "\t--save,     optional, save file or not, default true; select from [true/false]\n"
            "\t--help,     usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    const char *file_in_name = NULL;
    const char *file_out_name = NULL;

    bool need_save_output = true;

    const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"save", required_argument, NULL, 's'},
        {"help", no_argument, NULL, 'H'},
        {0, 0, 0, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            file_in_name = optarg;
            break;
        case 'o':
            file_out_name = optarg;
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

    if (optind < argc || argc < 2) {
        printf("unknown option %s\n", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    if (!file_in_name || !file_out_name) {
        XCAM_LOG_ERROR ("input/output path is NULL");
        return -1;
    }

    printf ("Description-----------\n");
    printf ("input image file:%s\n", file_in_name);
    printf ("output file :%s\n", file_out_name);
    printf ("need save file:%s\n", need_save_output ? "true" : "false");
    printf ("----------------------\n");

    SmartPtr<CVImageDeblurring> imageDeblurring = new CVImageDeblurring();
    SmartPtr<CVImageSharp> sharp = new CVImageSharp();
    cv::Mat blurred = cv::imread(file_in_name, CV_LOAD_IMAGE_COLOR);
    if (blurred.empty()) {
        XCAM_LOG_ERROR ("input file read error");
        return 0;
    }
    cv::Mat deblurred;
    cv::Mat kernel;
    imageDeblurring->blind_deblurring(blurred, deblurred, kernel);
    float input_sharp = sharp->measure_sharp(blurred);
    float output_sharp = sharp->measure_sharp(deblurred);
    assert(output_sharp > input_sharp);
    if (need_save_output) {
        cv::imwrite(file_out_name, deblurred);
    }
}

