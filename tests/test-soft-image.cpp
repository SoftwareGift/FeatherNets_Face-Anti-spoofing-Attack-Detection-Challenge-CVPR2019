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
#include "test_stream.h"
#include <soft/soft_video_buf_allocator.h>
#include <interface/blender.h>
#include <interface/geo_mapper.h>

#define MAP_WIDTH 3
#define MAP_HEIGHT 4

static PointFloat2 map_table[MAP_HEIGHT * MAP_WIDTH] = {
    {160.0f, 120.0f}, {480.0f, 120.0f}, {796.0f, 120.0f},
    {60.0f, 240.0f}, {480.0f, 240.0f}, {900.0f, 240.0f},
    {16.0f, 360.0f}, {480.0f, 360.0f}, {944.0f, 360.0f},
    {0.0f, 480.0f}, {480.0f, 480.0f}, {960.0f, 480.0f},
};

using namespace XCam;

enum SoftType {
    SoftTypeNone    = 0,
    SoftTypeBlender,
    SoftTypeRemap
};

class SoftStream
    : public Stream
{
public:
    explicit SoftStream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~SoftStream () {}

    virtual XCamReturn create_buf_pool (const VideoBufferInfo &info, uint32_t count);

private:
    XCAM_DEAD_COPY (SoftStream);
};
typedef std::vector<SmartPtr<SoftStream>> SoftStreams;

SoftStream::SoftStream (const char *file_name, uint32_t width, uint32_t height)
    :  Stream (file_name, width, height)
{
}

XCamReturn
SoftStream::create_buf_pool (const VideoBufferInfo &info, uint32_t count)
{
    SmartPtr<BufferPool> pool = new SoftVideoBufAllocator ();
    XCAM_ASSERT (pool.ptr ());

    pool->set_video_info (info);
    if (!pool->reserve (count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --type TYPE --input0 input.nv12 --input1 input1.nv12 --output output.nv12 ...\n"
            "\t--type              processing type, selected from: blend, remap\n"
            "\t--input0            input image(NV12)\n"
            "\t--input1            input image(NV12)\n"
            "\t--output            output image(NV12/MP4)\n"
            "\t--in-w              optional, input width, default: 1280\n"
            "\t--in-h              optional, input height, default: 800\n"
            "\t--out-w             optional, output width, default: 1280\n"
            "\t--out-h             optional, output height, default: 800\n"
            "\t--save              optional, save file or not, select from [true/false], default: true\n"
            "\t--loop              optional, how many loops need to run, default: 1\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    uint32_t input_width = 1280;
    uint32_t input_height = 800;
    uint32_t output_width = 1280;
    uint32_t output_height = 800;

    SoftStreams ins;
    SoftStreams outs;
    SoftType type = SoftTypeNone;

    int loop = 1;
    bool save_output = true;

    const struct option long_opts[] = {
        {"type", required_argument, NULL, 't'},
        {"input0", required_argument, NULL, 'i'},
        {"input1", required_argument, NULL, 'j'},
        {"output", required_argument, NULL, 'o'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"save", required_argument, NULL, 's'},
        {"loop", required_argument, NULL, 'l'},
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
            else {
                XCAM_LOG_ERROR ("unknown type:%s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'i':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SoftStream, ins, optarg);
            break;
        case 'j':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SoftStream, ins, optarg);
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SoftStream, outs, optarg);
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
        case 's':
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'l':
            loop = atoi(optarg);
            break;
        case 'e':
            usage (argv[0]);
            return 0;
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

    if (ins.empty () || outs.empty () ||
            !strlen (ins[0]->get_file_name ()) || !strlen (outs[0]->get_file_name ())) {
        XCAM_LOG_ERROR ("input or output file name was not set");
        usage (argv[0]);
        return -1;
    }

    for (uint32_t i = 0; i < ins.size (); ++i) {
        printf ("input%d file:\t\t%s\n", i, ins[i]->get_file_name ());
    }
    printf ("output file:\t\t%s\n", outs[0]->get_file_name ());
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("save output:\t\t%s\n", save_output ? "true" : "false");
    printf ("loop count:\t\t%d\n", loop);

    VideoBufferInfo in_info;
    in_info.init (V4L2_PIX_FMT_NV12, input_width, input_height);
    for (uint32_t i = 0; i < ins.size (); ++i) {
        ins[i]->set_buf_size (input_width, input_height);
        CHECK (ins[i]->create_buf_pool (in_info, 6), "create buffer pool failed");
        CHECK (ins[i]->open_reader ("rb"), "open input file(%s) failed", ins[i]->get_file_name ());
    }

    outs[0]->set_buf_size (output_width, output_height);
    if (save_output) {
        CHECK (outs[0]->estimate_file_format (), "%s: estimate file format failed", outs[0]->get_file_name ());
        CHECK (outs[0]->open_writer ("wb"), "open output file(%s) failed", outs[0]->get_file_name ());
    }

    switch (type) {
    case SoftTypeBlender: {
        CHECK_EXP (ins.size () == 2, "blender needs 2 input files.");
        SmartPtr<Blender> blender = Blender::create_soft_blender ();
        XCAM_ASSERT (blender.ptr ());
        blender->set_output_size (output_width, output_height);

        Rect area;
        area.pos_x = 0;
        area.pos_y = 0;
        area.width = output_width;
        area.height = output_height;
        blender->set_merge_window (area);
        area.pos_x = 0;
        area.pos_y = 0;
        area.width = input_width;
        area.height = input_height;
        blender->set_input_merge_area (area, 0);
        area.pos_x = 0;
        area.pos_y = 0;
        area.width = input_width;
        area.height = input_height;
        blender->set_input_merge_area (area, 1);

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        CHECK (ins[1]->read_buf(), "read buffer from file(%s) failed.", ins[1]->get_file_name ());
        for (int i = 0; i < loop; ++i) {
            CHECK (blender->blend (ins[0]->get_buf (), ins[1]->get_buf (), outs[0]->get_buf ()), "blend buffer failed");
            if (save_output)
                outs[0]->write_buf ();
            FPS_CALCULATION (soft-blend, XCAM_OBJ_DUR_FRAME_NUM);
        }
        break;
    }
    case SoftTypeRemap: {
        SmartPtr<GeoMapper> mapper = GeoMapper::create_soft_geo_mapper ();
        XCAM_ASSERT (mapper.ptr ());
        mapper->set_output_size (output_width, output_height);
        mapper->set_lookup_table (map_table, MAP_WIDTH, MAP_HEIGHT);
        //mapper->set_factors ((output_width - 1.0f) / (MAP_WIDTH - 1.0f), (output_height - 1.0f) / (MAP_HEIGHT - 1.0f));

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        for (int i = 0; i < loop; ++i) {
            CHECK (mapper->remap (ins[0]->get_buf (), outs[0]->get_buf ()), "remap buffer failed");
            if (save_output)
                outs[0]->write_buf ();
            FPS_CALCULATION (soft-remap, XCAM_OBJ_DUR_FRAME_NUM);
        }
        break;
    }
    default: {
        XCAM_LOG_ERROR ("unsupported type:%d", type);
        usage (argv[0]);
        return -1;
    }
    }

    return 0;
}
