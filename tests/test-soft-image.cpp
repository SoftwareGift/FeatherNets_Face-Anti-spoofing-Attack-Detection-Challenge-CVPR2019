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
#include "image_handler.h"
#include "image_file_handle.h"
#include "soft/soft_video_buf_allocator.h"
#include "interface/blender.h"
#include "interface/geo_mapper.h"
#include "interface/stitcher.h"
#include "calibration_parser.h"

#include <string>

#define RUN_N(statement, loop, msg, ...) \
    for (int i = 0; i < loop; ++i) {            \
        CHECK (statement, msg, ## __VA_ARGS__); \
    }

#define FISHEYE_CONFIG_PATH "./"

#define MAP_WIDTH 3
#define MAP_HEIGHT 4

static GeoData map_table[MAP_HEIGHT * MAP_WIDTH] = {
    {160.0f, 120.0f, 0.0f, 0.0f}, {480.0f, 120.0f, 0.0f, 0.0f}, {796.0f, 120.0f, 0.0f, 0.0f},
    {60.0f, 240.0f, 0.0f, 0.0f}, {480.0f, 240.0f, 0.0f, 0.0f}, {900.0f, 240.0f, 0.0f, 0.0f},
    {16.0f, 360.0f, 0.0f, 0.0f}, {480.0f, 360.0f, 0.0f, 0.0f}, {944.0f, 360.0f, 0.0f, 0.0f},
    {0.0f, 480.0f, 0.0f, 0.0f}, {480.0f, 480.0f, 0.0f, 0.0f}, {960.0f, 480.0f, 0.0f, 0.0f},
};

using namespace XCam;

enum SoftType {
    SoftTypeNone     = 0,
    SoftTypeBlender,
    SoftTypeRemap,
    SoftTypeStitch,
};

static int
parse_camera_info (const char *path, uint32_t idx, uint32_t out_w, uint32_t out_h, CameraInfo &info)
{
    static const char *instrinsic_names[] = {
        "intrinsic_camera_front.txt", "intrinsic_camera_right.txt",
        "intrinsic_camera_rear.txt", "intrinsic_camera_left.txt"
    };
    static const char *exstrinsic_names[] = {
        "extrinsic_camera_front.txt", "extrinsic_camera_right.txt",
        "extrinsic_camera_rear.txt", "extrinsic_camera_left.txt"
    };
    static const float viewpoints_range[] = {60.0f, 150.0f, 60.0f, 150.0f};

    char intrinsic_path[1024];
    char extrinsic_path[1024];
    snprintf (intrinsic_path, 1024, "%s/%s", path, instrinsic_names[idx]);
    snprintf (extrinsic_path, 1024, "%s/%s", path, exstrinsic_names[idx]);

    CalibrationParser parser;
    size_t file_size = 0;
    std::string context;
    FileHandle file_reader (intrinsic_path, "r");

    CHECK (file_reader.open (intrinsic_path, "r"), "open intrinsic file(%s) failed.", intrinsic_path );
    CHECK (file_reader.get_file_size (file_size), "read intrinsic file(%s) failed.", intrinsic_path);
    context.resize (file_size + 1);
    CHECK (file_reader.read_file (&context[0], file_size), "read intrinsic file(%s) failed.", intrinsic_path);
    file_reader.close ();

    CHECK (
        parser.parse_intrinsic_param (&context[0], info.calibration.intrinsic),
        "parse intrinsic params (%s)failed.", intrinsic_path);

    CHECK (file_reader.open (extrinsic_path, "r"), "open extrinsic file(%s) failed.", extrinsic_path );
    CHECK (file_reader.get_file_size (file_size), "read extrinsic file(%s) failed.", extrinsic_path);
    context.resize (file_size + 1);
    CHECK (file_reader.read_file (&context[0], file_size), "read extrinsic file(%s) failed.", extrinsic_path);
    file_reader.close ();
    CHECK (
        parser.parse_extrinsic_param (&context[0], info.calibration.extrinsic),
        "parse extrinsic params (%s)failed.", extrinsic_path);
    info.calibration.extrinsic.trans_x += TEST_CAMERA_POSITION_OFFSET_X;

    info.slice_view.width = viewpoints_range[idx] / 360.0f * out_w;
    info.slice_view.width = XCAM_ALIGN_UP (info.slice_view.width, 32);
    info.slice_view.height = out_h;
    info.slice_view.hori_angle_range = info.slice_view.width * 360.0f / (float)out_w;
    info.slice_view.hori_angle_start = (idx * 360.0f / 4.0f) - info.slice_view.hori_angle_range / 2.0f;
    info.slice_view.hori_angle_start = format_angle (info.slice_view.hori_angle_start);
    return 0;
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --type TYPE--input0 file0 --input1 file1 --output file\n"
            "\t--type              processing type, selected from: blend, remap, stitch, ...\n"
            "\t--input0            input image(NV12)\n"
            "\t--input1            input image(NV12)\n"
            "\t--input2            input image(NV12)\n"
            "\t--input3            input image(NV12)\n"
            "\t--output            output image(NV12)\n"
            "\t--in-w              optional, input width, default: 1920\n"
            "\t--in-h              optional, input height, default: 1080\n"
            "\t--out-w             optional, output width, default: 1920\n"
            "\t--out-h             optional, output width, default: 960\n"
            "\t--loop              optional, how many loops need to run, default: 1\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    char file_in0_name[256] = {'\0'};
    char file_in1_name[256] = {'\0'};
    char file_in2_name[256] = {'\0'};
    char file_in3_name[256] = {'\0'};
    char file_out_name[256] = {'\0'};
    //uint32_t input_format = V4L2_PIX_FMT_NV12;
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t output_width = 1920; //output_height * 2;
    uint32_t output_height = 960; //960;
    SoftType type = SoftTypeNone;
    int loop = 1;

    const struct option long_opts[] = {
        {"type", required_argument, NULL, 't'},
        {"input0", required_argument, NULL, 'i'},
        {"input1", required_argument, NULL, 'j'},
        {"input2", required_argument, NULL, 'k'},
        {"input3", required_argument, NULL, 'l'},
        {"output", required_argument, NULL, 'o'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"loop", required_argument, NULL, 'L'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 't':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "blend"))
                type = SoftTypeBlender;
            else if (!strcasecmp (optarg, "remap"))
                type = SoftTypeRemap;
            else if (!strcasecmp (optarg, "stitch"))
                type = SoftTypeStitch;
            else {
                XCAM_LOG_ERROR ("unknown type:%s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;

        case 'i':
            XCAM_ASSERT (optarg);
            strncpy(file_in0_name, optarg, sizeof (file_in0_name) - 1);
            break;
        case 'j':
            XCAM_ASSERT (optarg);
            strncpy(file_in1_name, optarg, sizeof (file_in1_name) - 1);
            break;
        case 'k':
            XCAM_ASSERT (optarg);
            strncpy(file_in2_name, optarg, sizeof (file_in2_name) - 1);
            break;
        case 'l':
            XCAM_ASSERT (optarg);
            strncpy(file_in3_name, optarg, sizeof (file_in3_name) - 1);
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
        case 'L':
            loop = atoi(optarg);
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

    if (SoftTypeNone == type) {
        XCAM_LOG_ERROR ("Type was not set");
        usage (argv[0]);
        return -1;
    }

    if (!strlen (file_in0_name) || !strlen (file_out_name)) {
        XCAM_LOG_ERROR ("input or output file name was not set");
        usage (argv[0]);
        return -1;
    }

    printf ("input0 file:\t\t%s\n", file_in0_name);
    printf ("input1 file:\t\t%s\n", file_in1_name);
    printf ("input2 file:\t\t%s\n", file_in2_name);
    printf ("input3 file:\t\t%s\n", file_in3_name);
    printf ("output file:\t\t%s\n", file_out_name);
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("loop count:\t\t%d\n", loop);

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
    if (!out_pool->reserve (4)) {
        XCAM_LOG_ERROR ("out-buffer pool reserve failed");
        return -1;
    }

    SmartPtr<VideoBuffer> in0 = in_pool->get_buffer (in_pool);
    SmartPtr<VideoBuffer> out = out_pool->get_buffer (out_pool);
    SmartPtr<VideoBuffer> in1, in2, in3;

    ImageFileHandle in0_file(file_in0_name, "rb");
    CHECK (in0_file.read_buf (in0), "read buffer from file(%s) failed.", file_in0_name);

    if (strlen (file_in1_name)) {
        in1 = in_pool->get_buffer (in_pool);
        ImageFileHandle in1_file(file_in1_name, "rb");
        CHECK (in1_file.read_buf (in1), "read buffer from file(%s) failed.", file_in1_name);
    }
    if (strlen (file_in2_name)) {
        in2 = in_pool->get_buffer (in_pool);
        ImageFileHandle in2_file(file_in2_name, "rb");
        CHECK (in2_file.read_buf (in2), "read buffer from file(%s) failed.", file_in2_name);
    }
    if (strlen (file_in3_name)) {
        in3 = in_pool->get_buffer (in_pool);
        ImageFileHandle in3_file(file_in3_name, "rb");
        CHECK (in3_file.read_buf (in3), "read buffer from file(%s) failed.", file_in3_name);
    }

    switch (type) {
    case SoftTypeBlender: {
        SmartPtr<Blender> blender = Blender::create_soft_blender ();
        XCAM_ASSERT (blender.ptr ());
        blender->set_output_size (output_width, output_height);
        Rect merge_window;
        merge_window.pos_x = 0;
        merge_window.pos_y = 0;
        merge_window.width = out_info.width;
        merge_window.height = out_info.height;
        blender->set_merge_window (merge_window);
        RUN_N (blender->blend (in0, in1, out), loop, "blend in0/in1 to out buffer failed.");
        break;
    }
    case SoftTypeRemap: {
        SmartPtr<GeoMapper> mapper = GeoMapper::create_soft_geo_mapper ();
        XCAM_ASSERT (mapper.ptr ());
        mapper->set_output_size (output_width, output_height);
        mapper->set_lookup_table (map_table, MAP_WIDTH, MAP_HEIGHT);
        //mapper->set_factors ((output_width - 1.0f) / (MAP_WIDTH - 1.0f), (output_height - 1.0f) / (MAP_HEIGHT - 1.0f));
        RUN_N (mapper->remap (in0, out), loop, "remap in0 to out buffer failed.");
        break;
    }
    case SoftTypeStitch: {
        SmartPtr<Stitcher> stitcher = Stitcher::create_soft_stitcher ();
        XCAM_ASSERT (stitcher.ptr ());
        XCAM_ASSERT (in0.ptr () && in1.ptr () && in2.ptr () && in3.ptr ());
        VideoBufferList in_buffers;
        in_buffers.push_back (in0);
        in_buffers.push_back (in1);
        in_buffers.push_back (in2);
        in_buffers.push_back (in3);
        CameraInfo cam_info[4];
        const char *fisheye_config_path = getenv ("FISHEYE_CONFIG_PATH");
        if (!fisheye_config_path)
            fisheye_config_path = FISHEYE_CONFIG_PATH;

        stitcher->set_camera_num (4);
        for (uint32_t i = 0; i < 4; ++i) {
            if (parse_camera_info (fisheye_config_path, i, output_width, output_height, cam_info[i]) != 0) {
                XCAM_LOG_ERROR ("parse fisheye dewarp info(idx:%d) failed.", i);
                return -1;
            }
            stitcher->set_camera_info (i, cam_info[i]);
        }
        BowlDataConfig bowl;
        bowl.wall_height = 3000.0f;
        bowl.ground_length = 2000.0f;
        //bowl.a = 5000.0f;
        //bowl.b = 3600.0f;
        //bowl.c = 3000.0f;
        bowl.angle_start = 0.0f;
        bowl.angle_end = 0.0f;
        stitcher->set_bowl_config (bowl);
        stitcher->set_output_size (output_width, output_height);
        RUN_N (stitcher->stitch_buffers (in_buffers, out), loop, "stitcher buffers to out buffer failed.");
        break;
    }

    default: {
        XCAM_LOG_ERROR ("unsupported type:%d", type);
        usage (argv[0]);
        return -1;
    }
    }

    ImageFileHandle out_file (file_out_name, "wb");
    CHECK (out_file.write_buf (out), "write buffer to file(%s) failed.", file_out_name);
    out_file.close ();
    return 0;
}
