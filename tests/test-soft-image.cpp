/*
 * test-soft-image.cpp - test soft image
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "test_common.h"
#include "test_inline.h"
#include "buffer_pool.h"
#include "interface/blender.h"
#include "image_handler.h"
#include "image_file_handle.h"
#include "soft/soft_video_buf_allocator.h"
#include "soft/soft_handler.h"

using namespace XCam;

void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input0 file0 --input1 file1 --output file\n"
            "\t--input0            input image(NV12)\n"
            "\t--input1            input image(NV12)\n"
            "\t--output            output image(NV12)\n"
            "\t--in-w              optional, input width, default: 1920\n"
            "\t--in-h              optional, input height, default: 1080\n"
            "\t--out-w             optional, output width, default: 1920\n"
            "\t--out-h             optional, output width, default: 960\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    char file_in0_name[256] = {'\0'};
    char file_in1_name[256] = {'\0'};
    char file_out_name[256] = {'\0'};
    //uint32_t input_format = V4L2_PIX_FMT_NV12;
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t output_width = 1920; //output_height * 2;
    uint32_t output_height = 960; //960;

    const struct option long_opts[] = {
        {"input0", required_argument, NULL, 'i'},
        {"input1", required_argument, NULL, 'j'},
        {"output", required_argument, NULL, 'o'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            XCAM_ASSERT (optarg);
            strncpy(file_in0_name, optarg, sizeof (file_in0_name) - 1);
            break;
        case 'j':
            XCAM_ASSERT (optarg);
            strncpy(file_in1_name, optarg, sizeof (file_in1_name) - 1);
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            strncpy(file_out_name, optarg, sizeof (file_out_name) - 1);
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

    if (!strlen (file_in0_name) || !strlen (file_in1_name) || !strlen (file_out_name)) {
        XCAM_LOG_ERROR ("input or output file name was not set");
        usage (argv[0]);
        return -1;
    }

    printf ("input0 file:\t\t%s\n", file_in0_name);
    printf ("input1 file:\t\t%s\n", file_in1_name);
    printf ("output file:\t\t%s\n", file_out_name);
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);

    VideoBufferInfo in_info;
    in_info.init (V4L2_PIX_FMT_NV12, input_width, input_height);
    SmartPtr<BufferPool> in_pool = new SoftVideoBufAllocator ();
    in_pool->set_video_info (in_info);
    if (!in_pool->reserve (4)) {
        XCAM_LOG_ERROR ("in-buffer pool reserve failed");
        return -1;
    }

    VideoBufferInfo out_info;
    out_info.init (V4L2_PIX_FMT_NV12, output_width, output_height);
    SmartPtr<BufferPool> out_pool = new SoftVideoBufAllocator ();
    out_pool->set_video_info (out_info);
    if (!out_pool->reserve (2)) {
        XCAM_LOG_ERROR ("out-buffer pool reserve failed");
        return -1;
    }

    SmartPtr<VideoBuffer> in0 = in_pool->get_buffer (in_pool);
    SmartPtr<VideoBuffer> in1 = in_pool->get_buffer (in_pool);
    SmartPtr<VideoBuffer> out = out_pool->get_buffer (out_pool);

    ImageFileHandle in0_file(file_in0_name, "rb");
    CHECK (in0_file.read_buf (in0), "read buffer from file(%s) failed.", file_in0_name);

    ImageFileHandle in1_file(file_in1_name, "rb");
    CHECK (in1_file.read_buf (in1), "read buffer from file(%s) failed.", file_in1_name);

    SmartPtr<Blender> blender = Blender::create_soft_blender ();
    XCAM_ASSERT (blender.ptr ());
    SmartPtr<SoftHandler> handler = blender.dynamic_cast_ptr<SoftHandler> ();
    blender->set_output_size (output_width, output_height);
    Rect merge_window;
    merge_window.pos_x = 0;
    merge_window.pos_y = 0;
    merge_window.width = out_info.width;
    merge_window.height = out_info.height;
    blender->set_merge_window (merge_window);
    CHECK (blender->blend (in0, in1, out), "blend in0/in1 to out buffer failed.");

    ImageFileHandle out_file (file_out_name, "wb");
    CHECK (out_file.write_buf (out), "write buffer to file(%s) failed.", file_out_name);
    out_file.close ();
    return 0;
}
