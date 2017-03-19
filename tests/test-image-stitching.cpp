/*
 * test-image-stitching.cpp - test image stitching
 *
 *  Copyright (c) 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "test_common.h"
#include <unistd.h>
#include <getopt.h>
#include "cl_device.h"
#include "cl_context.h"
#include "drm_display.h"
#include "image_file_handle.h"

#include "cl_fisheye_handler.h"
#include "cl_image_360_stitch.h"

#if HAVE_OPENCV
#include "cv_feature_match.h"
#endif

#define XCAM_STITCHING_DEBUG 0
#define XCAM_ALIGNED_WIDTH 16

using namespace XCam;

static CLStitchInfo
get_stitch_initial_info (uint32_t out_width, uint32_t out_height)
{
    CLStitchInfo stitch_info;

    stitch_info.output_width = out_width;
    stitch_info.output_height = out_height;

    stitch_info.merge_width[0] = 56;
    stitch_info.merge_width[1] = 56;

    stitch_info.crop[0].left = 96;
    stitch_info.crop[0].right = 96;
    stitch_info.crop[0].top = 0;
    stitch_info.crop[0].bottom = 0;
    stitch_info.crop[1].left = 96;
    stitch_info.crop[1].right = 96;
    stitch_info.crop[1].top = 0;
    stitch_info.crop[1].bottom = 0;

    stitch_info.fisheye_info[0].center_x = 480.0f;
    stitch_info.fisheye_info[0].center_y = 480.0f;
    stitch_info.fisheye_info[0].wide_angle = 202.8f;
    stitch_info.fisheye_info[0].radius = 480.0f;
    stitch_info.fisheye_info[0].rotate_angle = -90.0f;
    stitch_info.fisheye_info[1].center_x = 1440.0f;
    stitch_info.fisheye_info[1].center_y = 480.0f;
    stitch_info.fisheye_info[1].wide_angle = 202.8f;
    stitch_info.fisheye_info[1].radius = 480.0f;
    stitch_info.fisheye_info[1].rotate_angle = 89.4f;

    return stitch_info;
}

void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input file --output file\n"
            "\t--input       input image(NV12)\n"
            "\t--output      output image(NV12)\n"
            "\t--input-w     optional, input width, default: 1920\n"
            "\t--input-h     optional, input height, default: 1080\n"
            "\t--output-w    optional, output width, default: 1920\n"
            "\t--output-h    optional, output width, default: 960\n"
            "\t--loop        optional, how many loops need to run for performance test, default: 0\n"
            "\t--save        optional, save file or not, select from [true/false], default: true\n"
            "\t--scale-mode  optional, image scaling mode, select from [local/global], default: local\n"
            "\t--enable-seam optional, enable seam finder in blending area, default: no\n"
            "\t--help        usage\n",
            arg0);
}

static void
ensure_gpu_buffer_done (SmartPtr<BufferProxy> buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            int mem_idx = info.offsets [index] + i * info.strides [index] + line_bytes - 1;
            if (memory[mem_idx] == 1) {
                memory[mem_idx] = 1;
            }
        }
    }
    buf->unmap ();
}

int main (int argc, char *argv[])
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLContext> context;
    SmartPtr<DrmDisplay> display;
    SmartPtr<DrmBoBufferPool> buf_pool;
    SmartPtr<BufferProxy> read_buf;
    ImageFileHandle file_in, file_out;
    SmartPtr<DrmBoBuffer> input_buf, output_buf;
    VideoBufferInfo input_buf_info, output_buf_info;
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLImage360Stitch> image_360;

    uint32_t input_format = V4L2_PIX_FMT_NV12;
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t output_height = 960;
    uint32_t output_width = output_height * 2;

    int loop = 0;
    bool enable_seam = false;
    bool need_save_output = true;
    CLBlenderScaleMode scale_mode = CLBlenderScaleLocal;
    const char *file_in_name = NULL;
    const char *file_out_name = NULL;

    const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"input-w", required_argument, NULL, 'w'},
        {"input-h", required_argument, NULL, 'h'},
        {"output-w", required_argument, NULL, 'W'},
        {"output-h", required_argument, NULL, 'H'},
        {"loop", required_argument, NULL, 'l'},
        {"save", required_argument, NULL, 's'},
        {"scale-mode", required_argument, NULL, 'c'},
        {"enable-seam", no_argument, NULL, 'S'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            XCAM_ASSERT (optarg);
            file_in_name = optarg;
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            file_out_name = optarg;
            break;
        case 'w':
            input_width = atoi(optarg);
            break;
        case 'h':
            input_height = atoi(optarg);
            break;
        case 'W':
            output_width = atoi(optarg);
            break;
        case 'H':
            output_height = atoi(optarg);
            break;
        case 'l':
            loop = atoi(optarg);
            break;
        case 's':
            need_save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'c':
            if (!strcasecmp (optarg, "local"))
                scale_mode = CLBlenderScaleLocal;
            else if (!strcasecmp (optarg, "global"))
                scale_mode = CLBlenderScaleGlobal;
            else {
                XCAM_LOG_ERROR ("incorrect scaling mode");
                return -1;
            }
            break;
        case 'S':
            enable_seam = true;
            break;
        case 'e':
            usage (argv[0]);
            return -1;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value:%c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknown option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    if (!file_in_name || !file_out_name) {
        XCAM_LOG_ERROR ("input/output path is NULL");
        return -1;
    }

    output_width = XCAM_ALIGN_UP (output_width, XCAM_ALIGNED_WIDTH);
    output_height = XCAM_ALIGN_UP (output_height, XCAM_ALIGNED_WIDTH);
    if (output_width != output_height * 2) {
        XCAM_LOG_ERROR ("incorrect output size width:%d height:%d", output_width, output_height);
        return -1;
    }

    printf ("Description----------------\n");
    printf ("input file:\t%s\n", file_in_name);
    printf ("output file:\t%s\n", file_out_name);
    printf ("input width:\t%d\n", input_width);
    printf ("input height:\t%d\n", input_height);
    printf ("output width:\t%d\n", output_width);
    printf ("output height:\t%d\n", output_height);
    printf ("loop count:\t%d\n", loop);
    printf ("save file:\t%s\n", need_save_output ? "true" : "false");
    printf ("scale mode:\t%s\n", scale_mode == CLBlenderScaleLocal? "local" : "global");
    printf ("seam mask:\t%s\n", enable_seam ? "true" : "false");
    printf ("---------------------------\n");

    context = CLDevice::instance ()->get_context ();
    image_360 = create_image_360_stitch (context, enable_seam, scale_mode).dynamic_cast_ptr<CLImage360Stitch> ();
    XCAM_ASSERT (image_360.ptr ());
    CLStitchInfo stitch_info = get_stitch_initial_info (output_width, output_height);
    image_360->init_stitch_info (stitch_info);

    input_buf_info.init (input_format, input_width, input_height);
    output_buf_info.init (input_format, output_width, output_height);
    display = DrmDisplay::instance ();
    buf_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (buf_pool.ptr ());
    buf_pool->set_video_info (input_buf_info);
    if (!buf_pool->reserve (2)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    ret = file_in.open (file_in_name, "rb");
    CHECK (ret, "open %s failed", file_in_name);

#if HAVE_OPENCV
    init_opencv_ocl (context);

    cv::VideoWriter writer;
    if (need_save_output) {
        cv::Size dst_size = cv::Size (output_width, output_height);
        if (!writer.open (file_out_name, CV_FOURCC('X', '2', '6', '4'), 30, dst_size)) {
            XCAM_LOG_ERROR ("open file %s failed", file_out_name);
            return -1;
        }
    }
#endif

    int i = 0;
    do {
        input_buf = buf_pool->get_buffer (buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
        XCAM_ASSERT (input_buf.ptr ());
        read_buf = input_buf;
        ret = file_in.read_buf (read_buf);
        if (ret == XCAM_RETURN_BYPASS)
            break;
        if (ret == XCAM_RETURN_ERROR_FILE) {
            XCAM_LOG_ERROR ("read buffer from %s failed", file_in_name);
            return -1;
        }

        ret = image_360->execute (input_buf, output_buf);
        CHECK (ret, "image_360 stitch execute failed");

        if (need_save_output) {
#if HAVE_OPENCV
            cv::Mat out_mat;
            convert_to_mat (context, output_buf, out_mat);
            writer.write (out_mat);
#endif
        } else {
            ensure_gpu_buffer_done (output_buf);
        }

        FPS_CALCULATION (image_stitching, 30);
        ++i;
    } while (true);

    return 0;
}

