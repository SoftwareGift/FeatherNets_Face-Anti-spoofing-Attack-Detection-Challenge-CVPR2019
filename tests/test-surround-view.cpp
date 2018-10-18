/*
 * test-surround-view.cpp - test surround view
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
#include <interface/geo_mapper.h>
#include <interface/stitcher.h>
#include <calibration_parser.h>
#include <soft/soft_video_buf_allocator.h>
#if HAVE_GLES
#include <gles/gl_video_buffer.h>
#include <gles/egl/egl_base.h>
#endif

using namespace XCam;

enum FrameMode {
    FrameSingle = 0,
    FrameMulti
};

enum SVModule {
    SVModuleNone    = 0,
    SVModuleSoft,
    SVModuleGLES
};

enum SVOutIdx {
    IdxStitch    = 0,
    IdxTopView,
    IdxCount
};

class SVStream
    : public Stream
{
public:
    explicit SVStream (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    virtual ~SVStream () {}

    void set_module (SVModule module) {
        XCAM_ASSERT (module != SVModuleNone);
        _module = module;
    }

    void set_mapper (const SmartPtr<GeoMapper> &mapper) {
        XCAM_ASSERT (mapper.ptr ());
        _mapper = mapper;
    }
    const SmartPtr<GeoMapper> &get_mapper () {
        return _mapper;
    }

    virtual XCamReturn create_buf_pool (const VideoBufferInfo &info, uint32_t count);

private:
    XCAM_DEAD_COPY (SVStream);

private:
    SVModule               _module;
    SmartPtr<GeoMapper>    _mapper;
};
typedef std::vector<SmartPtr<SVStream>> SVStreams;

SVStream::SVStream (const char *file_name, uint32_t width, uint32_t height)
    :  Stream (file_name, width, height)
    , _module (SVModuleNone)
{
}

XCamReturn
SVStream::create_buf_pool (const VideoBufferInfo &info, uint32_t count)
{
    XCAM_FAIL_RETURN (
        ERROR, _module != SVModuleNone, XCAM_RETURN_ERROR_PARAM,
        "invalid module, please set module first");

    SmartPtr<BufferPool> pool;
    if (_module == SVModuleSoft) {
        pool = new SoftVideoBufAllocator (info);
    } else if (_module == SVModuleGLES) {
#if HAVE_GLES
        pool = new GLVideoBufferPool (info);
#endif
    }
    XCAM_ASSERT (pool.ptr ());

    if (!pool->reserve (count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<Stitcher>
create_stitcher (SVModule module)
{
    SmartPtr<Stitcher> stitcher;

    if (module == SVModuleSoft) {
        stitcher = Stitcher::create_soft_stitcher ();
    } else if (module == SVModuleGLES) {
#if HAVE_GLES
        stitcher = Stitcher::create_gl_stitcher ();
#endif
    }
    XCAM_ASSERT (stitcher.ptr ());

    return stitcher;
}

static int
parse_camera_info (const char *path, uint32_t idx, CameraInfo &info, uint32_t camera_count)
{
    static const char *instrinsic_names[] = {
        "intrinsic_camera_front.txt", "intrinsic_camera_right.txt",
        "intrinsic_camera_rear.txt", "intrinsic_camera_left.txt"
    };
    static const char *exstrinsic_names[] = {
        "extrinsic_camera_front.txt", "extrinsic_camera_right.txt",
        "extrinsic_camera_rear.txt", "extrinsic_camera_left.txt"
    };
    static const float viewpoints_range[] = {64.0f, 160.0f, 64.0f, 160.0f};

    char intrinsic_path[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    char extrinsic_path[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    snprintf (intrinsic_path, XCAM_TEST_MAX_STR_SIZE, "%s/%s", path, instrinsic_names[idx]);
    snprintf (extrinsic_path, XCAM_TEST_MAX_STR_SIZE, "%s/%s", path, exstrinsic_names[idx]);

    CalibrationParser parser;
    CHECK (
        parser.parse_intrinsic_file (intrinsic_path, info.calibration.intrinsic),
        "parse intrinsic params (%s)failed.", intrinsic_path);

    CHECK (
        parser.parse_extrinsic_file (extrinsic_path, info.calibration.extrinsic),
        "parse extrinsic params (%s)failed.", extrinsic_path);
    info.calibration.extrinsic.trans_x += TEST_CAMERA_POSITION_OFFSET_X;

    info.angle_range = viewpoints_range[idx];
    info.round_angle_start = (idx * 360.0f / camera_count) - info.angle_range / 2.0f;
    return 0;
}

static void
combine_name (const char *orig_name, const char *embedded_str, char *new_name)
{
    const char *dir_delimiter = strrchr (orig_name, '/');

    if (dir_delimiter) {
        std::string path (orig_name, dir_delimiter - orig_name + 1);
        XCAM_ASSERT (path.c_str ());
        snprintf (new_name, XCAM_TEST_MAX_STR_SIZE, "%s%s_%s", path.c_str (), embedded_str, dir_delimiter + 1);
    } else {
        snprintf (new_name, XCAM_TEST_MAX_STR_SIZE, "%s_%s", embedded_str, orig_name);
    }
}

static void
add_stream (SVStreams &streams, const char *stream_name, uint32_t width, uint32_t height)
{
    char file_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    combine_name (streams[0]->get_file_name (), stream_name, file_name);

    SmartPtr<SVStream> stream = new SVStream (file_name, width, height);
    XCAM_ASSERT (stream.ptr ());
    streams.push_back (stream);
}

static void
write_in_image (const SVStreams &ins, uint32_t frame_num)
{
#if (XCAM_TEST_STREAM_DEBUG) && (XCAM_TEST_OPENCV)
    char frame_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    std::snprintf (frame_str, XCAM_TEST_MAX_STR_SIZE, "frame:%d", frame_num);

    char img_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    char idx_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    for (uint32_t i = 0; i < ins.size (); ++i) {
        std::snprintf (idx_str, XCAM_TEST_MAX_STR_SIZE, "idx:%d", i);
        std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "orig_fisheye_%d_%d.jpg", frame_num, i);
        ins[i]->debug_write_image (img_name, frame_str, idx_str);
    }
#else
    XCAM_UNUSED (ins);
    XCAM_UNUSED (frame_num);
#endif
}

static void
write_out_image (const SmartPtr<SVStream> &out, uint32_t frame_num)
{
#if !XCAM_TEST_STREAM_DEBUG
    XCAM_UNUSED (frame_num);
    out->write_buf ();
#else
    char frame_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    std::snprintf (frame_str, XCAM_TEST_MAX_STR_SIZE, "frame:%d", frame_num);
    out->write_buf (frame_str);

#if XCAM_TEST_OPENCV
    char img_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "%s_%d.jpg", out->get_file_name (), frame_num);
    out->debug_write_image (img_name, frame_str);
#endif
#endif
}

static XCamReturn
create_topview_mapper (
    const SmartPtr<Stitcher> &stitcher, const SmartPtr<SVStream> &stitch,
    const SmartPtr<SVStream> &topview, SVModule module)
{
    BowlModel bowl_model (stitcher->get_bowl_config (), stitch->get_width (), stitch->get_height ());
    BowlModel::PointMap points;

    float length_mm = 0.0f, width_mm = 0.0f;
    bowl_model.get_max_topview_area_mm (length_mm, width_mm);
    XCAM_LOG_INFO ("Max Topview Area (L%.2fmm, W%.2fmm)", length_mm, width_mm);

    bowl_model.get_topview_rect_map (points, topview->get_width (), topview->get_height (), length_mm, width_mm);
    SmartPtr<GeoMapper> mapper;
    if (module == SVModuleSoft) {
        mapper = GeoMapper::create_soft_geo_mapper ();
    } else if (module == SVModuleGLES) {
#if HAVE_GLES
        mapper = GeoMapper::create_gl_geo_mapper ();
#endif
    }
    XCAM_ASSERT (mapper.ptr ());

    mapper->set_output_size (topview->get_width (), topview->get_height ());
    mapper->set_lookup_table (points.data (), topview->get_width (), topview->get_height ());
    topview->set_mapper (mapper);

    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
remap_topview_buf (const SmartPtr<SVStream> &stitch, const SmartPtr<SVStream> &topview)
{
    const SmartPtr<GeoMapper> mapper = topview->get_mapper();
    XCAM_ASSERT (mapper.ptr ());

    XCamReturn ret = mapper->remap (stitch->get_buf (), topview->get_buf ());
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("remap stitched image to topview failed.");
        return ret;
    }

    return XCAM_RETURN_NO_ERROR;
}

static void
write_image (
    const SVStreams &ins, const SVStreams &outs, bool save_output, bool save_topview)
{
    static uint32_t frame_num = 0;

    write_in_image (ins, frame_num);

    if (save_output)
        write_out_image (outs[IdxStitch], frame_num);

    if (save_topview) {
        remap_topview_buf (outs[IdxStitch], outs[IdxTopView]);
        write_out_image (outs[IdxTopView], frame_num);
    }

    frame_num++;
}

static int
single_frame (
    const SmartPtr<Stitcher> &stitcher,
    const SVStreams &ins, const SVStreams &outs,
    bool save_output, bool save_topview, int loop)
{
    for (uint32_t i = 0; i < ins.size (); ++i) {
        CHECK (ins[i]->rewind (), "rewind buffer from file(%s) failed", ins[i]->get_file_name ());
    }

    VideoBufferList in_buffers;
    for (uint32_t i = 0; i < ins.size (); ++i) {
        XCamReturn ret = ins[i]->read_buf ();
        CHECK_EXP (ret == XCAM_RETURN_NO_ERROR, "read buffer from file(%s) failed.", ins[i]->get_file_name ());

        XCAM_ASSERT (ins[i]->get_buf ().ptr ());
        in_buffers.push_back (ins[i]->get_buf ());
    }

    while (loop--) {
        CHECK (stitcher->stitch_buffers (in_buffers, outs[IdxStitch]->get_buf ()), "stitch buffer failed.");

        if (save_output || save_topview)
            write_image (ins, outs, save_output, save_topview);

        FPS_CALCULATION (surround-view, XCAM_OBJ_DUR_FRAME_NUM);
    }

    return 0;
}

static int
multi_frame (
    const SmartPtr<Stitcher> &stitcher,
    const SVStreams &ins, const SVStreams &outs,
    bool save_output, bool save_topview, int loop)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    VideoBufferList in_buffers;
    while (loop--) {
        for (uint32_t i = 0; i < ins.size (); ++i) {
            CHECK (ins[i]->rewind (), "rewind buffer from file(%s) failed", ins[i]->get_file_name ());
        }

        do {
            in_buffers.clear ();

            for (uint32_t i = 0; i < ins.size (); ++i) {
                ret = ins[i]->read_buf();
                if (ret == XCAM_RETURN_BYPASS)
                    break;
                CHECK (ret, "read buffer from file(%s) failed.", ins[i]->get_file_name ());

                in_buffers.push_back (ins[i]->get_buf ());
            }
            if (ret == XCAM_RETURN_BYPASS)
                break;

            CHECK (
                stitcher->stitch_buffers (in_buffers, outs[IdxStitch]->get_buf ()),
                "stitch buffer failed.");

            if (save_output || save_topview)
                write_image (ins, outs, save_output, save_topview);

            FPS_CALCULATION (surround-view, XCAM_OBJ_DUR_FRAME_NUM);
        } while (true);
    }

    return 0;
}

static int
run_stitcher (
    const SmartPtr<Stitcher> &stitcher,
    const SVStreams &ins, const SVStreams &outs,
    FrameMode frame_mode, bool save_output, bool save_topview, int loop)
{
    CHECK (check_streams<SVStreams> (ins), "invalid input streams");
    CHECK (check_streams<SVStreams> (outs), "invalid output streams");

    int ret = -1;
    if (frame_mode == FrameSingle)
        ret = single_frame (stitcher, ins, outs, save_output, save_topview, loop);
    else if (frame_mode == FrameMulti)
        ret = multi_frame (stitcher, ins, outs, save_output, save_topview, loop);
    else
        XCAM_LOG_ERROR ("invalid frame mode: %d", frame_mode);

    return ret;
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --module MODULE --input0 input.nv12 --input1 input1.nv12 --input2 input2.nv12 ...\n"
            "\t--module            processing module, selected from: soft, gles\n"
            "\t--                  read calibration files from exported path $FISHEYE_CONFIG_PATH\n"
            "\t--input0            input image(NV12)\n"
            "\t--input1            input image(NV12)\n"
            "\t--input2            input image(NV12)\n"
            "\t--input3            input image(NV12)\n"
            "\t--output            output image(NV12/MP4)\n"
            "\t--in-w              optional, input width, default: 1280\n"
            "\t--in-h              optional, input height, default: 800\n"
            "\t--out-w             optional, output width, default: 1920\n"
            "\t--out-h             optional, output height, default: 640\n"
            "\t--topview-w         optional, output width, default: 1280\n"
            "\t--topview-h         optional, output height, default: 720\n"
            "\t--scale-mode        optional, scaling mode for geometric mapping,\n"
            "\t                    select from [singleconst/dualconst/dualcurve], default: singleconst\n"
            "\t--frame-mode        optional, times of buffer reading, select from [single/multi], default: multi\n"
            "\t--save              optional, save file or not, select from [true/false], default: true\n"
            "\t--save-topview      optional, save top view video, select from [true/false], default: false\n"
            "\t--loop              optional, how many loops need to run, default: 1\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    uint32_t input_width = 1280;
    uint32_t input_height = 800;
    uint32_t output_width = 1920;
    uint32_t output_height = 640;
    uint32_t topview_width = 1280;
    uint32_t topview_height = 720;

    SVStreams ins;
    SVStreams outs;

    FrameMode frame_mode = FrameMulti;
    SVModule module = SVModuleNone;
    GeoMapScaleMode scale_mode = ScaleSingleConst;

    int loop = 1;
    bool save_output = true;
    bool save_topview = false;

    const struct option long_opts[] = {
        {"module", required_argument, NULL, 'm'},
        {"input0", required_argument, NULL, 'i'},
        {"input1", required_argument, NULL, 'j'},
        {"input2", required_argument, NULL, 'k'},
        {"input3", required_argument, NULL, 'l'},
        {"output", required_argument, NULL, 'o'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"topview-w", required_argument, NULL, 'P'},
        {"topview-h", required_argument, NULL, 'V'},
        {"scale-mode", required_argument, NULL, 'S'},
        {"frame-mode", required_argument, NULL, 'f'},
        {"save", required_argument, NULL, 's'},
        {"save-topview", required_argument, NULL, 't'},
        {"loop", required_argument, NULL, 'L'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'm':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "soft"))
                module = SVModuleSoft;
            else if (!strcasecmp (optarg, "gles")) {
                module = SVModuleGLES;
            } else {
                XCAM_LOG_ERROR ("unknown module:%s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'i':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SVStream, ins, optarg);
            break;
        case 'j':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SVStream, ins, optarg);
            break;
        case 'k':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SVStream, ins, optarg);
            break;
        case 'l':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SVStream, ins, optarg);
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            PUSH_STREAM (SVStream, outs, optarg);
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
        case 'P':
            topview_width = atoi(optarg);
            break;
        case 'V':
            topview_height = atoi(optarg);
            break;
        case 'S':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "singleconst"))
                scale_mode = ScaleSingleConst;
            else if (!strcasecmp (optarg, "dualconst"))
                scale_mode = ScaleDualConst;
            else if (!strcasecmp (optarg, "dualcurve"))
                scale_mode = ScaleDualCurve;
            else {
                XCAM_LOG_ERROR ("GeoMapScaleMode unknown mode: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'f':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "single"))
                frame_mode = FrameSingle;
            else if (!strcasecmp (optarg, "multi"))
                frame_mode = FrameMulti;
            else {
                XCAM_LOG_ERROR ("FrameMode unknown mode: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 's':
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 't':
            save_topview = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'L':
            loop = atoi(optarg);
            break;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value: %c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknown option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    CHECK_EXP (ins.size () == 4, "surrond view needs 4 input streams");
    for (uint32_t i = 0; i < ins.size (); ++i) {
        CHECK_EXP (ins[i].ptr (), "input stream is NULL, index:%d", i);
        CHECK_EXP (strlen (ins[i]->get_file_name ()), "input file name was not set, index:%d", i);
    }

    CHECK_EXP (outs.size () == 1 && outs[IdxStitch].ptr (), "surrond view needs 1 output stream");
    CHECK_EXP (strlen (outs[IdxStitch]->get_file_name ()), "output file name was not set");

    for (uint32_t i = 0; i < ins.size (); ++i) {
        printf ("input%d file:\t\t%s\n", i, ins[i]->get_file_name ());
    }
    printf ("output file:\t\t%s\n", outs[IdxStitch]->get_file_name ());
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("topview width:\t\t%d\n", topview_width);
    printf ("topview height:\t\t%d\n", topview_height);
    printf ("scaling mode:\t\t%s\n", (scale_mode == ScaleSingleConst) ? "singleconst" :
            ((scale_mode == ScaleDualConst) ? "dualconst" : "dualcurve"));
    printf ("frame mode:\t\t%s\n", (frame_mode == FrameSingle) ? "singleframe" : "multiframe");
    printf ("save output:\t\t%s\n", save_output ? "true" : "false");
    printf ("save topview:\t\t%s\n", save_topview ? "true" : "false");
    printf ("loop count:\t\t%d\n", loop);

    if (module == SVModuleGLES) {
#if !HAVE_GLES
        XCAM_LOG_ERROR ("GLES module unsupported");
        return -1;
#endif
    }

#if HAVE_GLES
    SmartPtr<EGLBase> egl;
    if (module == SVModuleGLES) {
        egl = new EGLBase ();
        XCAM_ASSERT (egl.ptr ());
        XCAM_FAIL_RETURN (ERROR, egl->init (), -1, "init EGL failed");
    }
#endif

    VideoBufferInfo in_info;
    in_info.init (V4L2_PIX_FMT_NV12, input_width, input_height);
    for (uint32_t i = 0; i < ins.size (); ++i) {
        ins[i]->set_module (module);
        ins[i]->set_buf_size (input_width, input_height);
        CHECK (ins[i]->create_buf_pool (in_info, 6), "create buffer pool failed");
        CHECK (ins[i]->open_reader ("rb"), "open input file(%s) failed", ins[i]->get_file_name ());
    }

    outs[IdxStitch]->set_buf_size (output_width, output_height);
    if (save_output) {
        CHECK (outs[IdxStitch]->estimate_file_format (),
            "%s: estimate file format failed", outs[IdxStitch]->get_file_name ());
        CHECK (outs[IdxStitch]->open_writer ("wb"), "open output file(%s) failed", outs[IdxStitch]->get_file_name ());
    }

    SmartPtr<Stitcher> stitcher = create_stitcher (module);
    XCAM_ASSERT (stitcher.ptr ());

    CameraInfo cam_info[4];
    const char *fisheye_config_path = getenv (FISHEYE_CONFIG_ENV_VAR);
    if (!fisheye_config_path)
        fisheye_config_path = FISHEYE_CONFIG_PATH;
    XCAM_LOG_INFO ("calibration config path:%s", XCAM_STR (fisheye_config_path));

    uint32_t camera_count = ins.size ();
    for (uint32_t i = 0; i < camera_count; ++i) {
        if (parse_camera_info (fisheye_config_path, i, cam_info[i], camera_count) != 0) {
            XCAM_LOG_ERROR ("parse fisheye dewarp info(idx:%d) failed.", i);
            return -1;
        }
    }

    PointFloat3 bowl_coord_offset;
    centralize_bowl_coord_from_cameras (
        cam_info[0].calibration.extrinsic, cam_info[1].calibration.extrinsic,
        cam_info[2].calibration.extrinsic, cam_info[3].calibration.extrinsic,
        bowl_coord_offset);

    stitcher->set_camera_num (camera_count);
    for (uint32_t i = 0; i < camera_count; ++i) {
        stitcher->set_camera_info (i, cam_info[i]);
    }

    BowlDataConfig bowl;
    bowl.wall_height = 3000.0f;
    bowl.ground_length = 2000.0f;
    bowl.angle_start = 0.0f;
    bowl.angle_end = 360.0f;
    stitcher->set_bowl_config (bowl);
    stitcher->set_output_size (output_width, output_height);
    stitcher->set_scale_mode (scale_mode);

    if (save_topview) {
        add_stream (outs, "topview", topview_width, topview_height);
        XCAM_ASSERT (outs.size () >= IdxCount);

        CHECK (outs[IdxTopView]->estimate_file_format (),
            "%s: estimate file format failed", outs[IdxTopView]->get_file_name ());
        CHECK (outs[IdxTopView]->open_writer ("wb"), "open output file(%s) failed", outs[IdxTopView]->get_file_name ());

        create_topview_mapper (stitcher, outs[IdxStitch], outs[IdxTopView], module);
    }

    CHECK_EXP (
        run_stitcher (stitcher, ins, outs, frame_mode, save_output, save_topview, loop) == 0,
        "run stitcher failed");

    return 0;
}
