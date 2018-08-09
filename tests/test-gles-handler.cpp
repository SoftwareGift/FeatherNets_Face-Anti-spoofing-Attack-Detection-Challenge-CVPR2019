/*
 * test-gles-handler.cpp - test gles handler
 *
 *  Copyright (c) 2018 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "test_common.h"
#include "test_stream.h"

#include <gles/gl_video_buffer.h>
#include <gles/egl/egl_base.h>
#include <gles/gl_copy_handler.h>
#include <gles/gl_geomap_handler.h>

using namespace XCam;

enum GLType {
    GLTypeNone    = 0,
    GLTypeCopy,
    GLTypeRemap
};

class GLStream
    : public Stream
{
public:
    explicit GLStream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~GLStream () {}

    virtual XCamReturn create_buf_pool (const VideoBufferInfo &info, uint32_t count);
};

typedef std::vector<SmartPtr<GLStream>> GLStreams;

GLStream::GLStream (const char *file_name, uint32_t width, uint32_t height)
    : Stream (file_name, width, height)
{
}

XCamReturn
GLStream::create_buf_pool (const VideoBufferInfo &info, uint32_t count)
{
    SmartPtr<GLVideoBufferPool> pool = new GLVideoBufferPool (info);
    XCAM_ASSERT (pool.ptr ());

    if (!pool->reserve (count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

static void
calc_hor_flip_table (uint32_t width, uint32_t height, PointFloat2 *&map_table)
{
    XCAM_ASSERT (map_table);

    float lut_size[2] = {8, 8};
    for (uint32_t i = 0; i < height; ++i) {
        PointFloat2 *line = &map_table[i * width];
        for (uint32_t j = 0; j < width; j++) {
            line[j].x = (width - j) * lut_size[0];
            line[j].y = i * lut_size[1];
        }
    }
}

static void usage (const char *arg0)
{
    printf ("Usage:\n"
            "%s --input0 input.nv12 --output output.nv12 ...\n"
            "\t--type              processing type, selected from: copy, remap\n"
            "\t--input0            input image(NV12)\n"
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

int main (int argc, char **argv)
{
    uint32_t input_width = 1280;
    uint32_t input_height = 800;
    uint32_t output_width = 1280;
    uint32_t output_height = 800;

    GLStreams ins;
    GLStreams outs;
    GLType type = GLTypeNone;

    int loop = 1;
    bool save_output = true;

    const struct option long_opts[] = {
        {"type", required_argument, NULL, 't'},
        {"input0", required_argument, NULL, 'i'},
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
            if (!strcasecmp (optarg, "copy"))
                type = GLTypeCopy;
            else if (!strcasecmp (optarg, "remap"))
                type = GLTypeRemap;
            else {
                XCAM_LOG_ERROR ("unknown type:%s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'i':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (GLStream, ins, optarg);
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (GLStream, outs, optarg);
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

    if (ins.empty () || outs.empty ()) {
        XCAM_LOG_ERROR ("input or output stream is empty");
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

    SmartPtr<EGLBase> egl = new EGLBase ();
    XCAM_FAIL_RETURN (ERROR, egl->init (), -1, "init EGL failed");

    VideoBufferInfo in_info;
    in_info.init (V4L2_PIX_FMT_NV12, input_width, input_height);
    for (uint32_t i = 0; i < ins.size (); ++i) {
        ins[i]->set_buf_size (input_width, input_height);
        CHECK (ins[i]->create_buf_pool (in_info, XCAM_GL_RESERVED_BUF_COUNT), "create buffer pool failed");
        CHECK (ins[i]->open_reader ("rb"), "open input file(%s) failed", ins[i]->get_file_name ());
    }

    VideoBufferInfo out_info;
    out_info.init (V4L2_PIX_FMT_NV12, output_width, output_height);
    outs[0]->set_buf_size (output_width, output_height);
    if (save_output) {
        CHECK (outs[0]->estimate_file_format (), "%s: estimate file format failed", outs[0]->get_file_name ());
        CHECK (outs[0]->open_writer ("wb"), "open output file(%s) failed", outs[0]->get_file_name ());
    }

    switch (type) {
    case GLTypeCopy: {
        SmartPtr<GLCopyHandler> copyer = new GLCopyHandler ();
        XCAM_ASSERT (copyer.ptr ());

        Rect in_area = Rect (0, 0, output_width, output_height);
        Rect out_area = in_area;
        copyer->set_copy_area (0, in_area, out_area);
        copyer->set_out_video_info (out_info);

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        for (int i = 0; i < loop; ++i) {
            CHECK (copyer->copy (ins[0]->get_buf (), outs[0]->get_buf ()), "copy buffer failed");
            if (save_output)
                outs[0]->write_buf ();
            FPS_CALCULATION (gl-copy, XCAM_OBJ_DUR_FRAME_NUM);
        }
        break;
    }
    case GLTypeRemap: {
        SmartPtr<GLGeoMapHandler> mapper = new GLGeoMapHandler ();
        XCAM_ASSERT (mapper.ptr ());
        mapper->set_output_size (output_width, output_height);

        uint32_t lut_width = XCAM_ALIGN_UP (output_width, 8) / 8;
        uint32_t lut_height = XCAM_ALIGN_UP (output_height, 8) / 8;
        PointFloat2 *map_table = new PointFloat2[lut_width * lut_height];
        calc_hor_flip_table (lut_width, lut_height, map_table);
        mapper->set_lookup_table (map_table, lut_width, lut_height);

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        for (int i = 0; i < loop; ++i) {
            CHECK (mapper->remap (ins[0]->get_buf (), outs[0]->get_buf ()), "remap buffer failed");
            if (save_output)
                outs[0]->write_buf ();
            FPS_CALCULATION (gl-remap, XCAM_OBJ_DUR_FRAME_NUM);
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
