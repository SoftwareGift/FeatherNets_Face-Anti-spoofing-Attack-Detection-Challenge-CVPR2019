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
#include <buffer_pool.h>
#include <image_handler.h>
#include <image_file_handle.h>
#include <soft/soft_video_buf_allocator.h>
#include <interface/blender.h>
#include <interface/geo_mapper.h>
#include <interface/stitcher.h>
#include <calibration_parser.h>
#include <string>
#include <cstring>

#if (!defined(ANDROID) && (HAVE_OPENCV))
#include <ocl/cv_base_class.h>
#endif

#define XCAM_TEST_SOFT_IMAGE_DEBUG 0

#if (!defined(ANDROID) && (HAVE_OPENCV))
#define XCAM_TEST_OPENCV 1
#else
#define XCAM_TEST_OPENCV 0
#endif

#define XCAM_TEST_MAX_STR_SIZE 1024

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
    SoftTypeNone     = 0,
    SoftTypeBlender,
    SoftTypeRemap,
    SoftTypeStitch,
};

#define RUN_N(statement, loop, msg, ...) \
    for (int i = 0; i < loop; ++i) {                          \
        CHECK (statement, msg, ## __VA_ARGS__);               \
        FPS_CALCULATION (soft-image, XCAM_OBJ_DUR_FRAME_NUM); \
    }

#define ADD_ENELEMT(elements, file_name) \
    {                                                                \
        SmartPtr<SoftElement> element = new SoftElement (file_name); \
        elements.push_back (element);                                \
    }

#if XCAM_TEST_OPENCV
const static cv::Scalar color = cv::Scalar (0, 0, 255);
const static int fontFace = cv::FONT_HERSHEY_COMPLEX;
#endif

class SoftElement {
public:
    explicit SoftElement (const char *file_name = NULL, uint32_t width = 0, uint32_t height = 0);
    ~SoftElement ();

    void set_buf_size (uint32_t width, uint32_t height);
    uint32_t get_width () const {
        return _width;
    }
    uint32_t get_height () const {
        return _height;
    }

    const char *get_file_name () const {
        return _file_name;
    }

    SmartPtr<VideoBuffer> &get_buf () {
        return _buf;
    }

    void set_mapper (SmartPtr<GeoMapper> mapper) {
        _mapper = mapper;
    }
    SmartPtr<GeoMapper> get_mapper () {
        return _mapper;
    }

    XCamReturn open_file (const char *option);
    XCamReturn close_file ();
    XCamReturn rewind_file ();

    XCamReturn read_buf ();
    XCamReturn write_buf ();

    XCamReturn create_buf_pool (const VideoBufferInfo &info, uint32_t count);

#if XCAM_TEST_OPENCV
    XCamReturn cv_open_writer ();
    void cv_write_image (char *img_name, char *frame_str, char *idx_str = NULL);
#endif

private:
    XCAM_DEAD_COPY (SoftElement);

private:
    char                 *_file_name;
    uint32_t              _width;
    uint32_t              _height;
    SmartPtr<VideoBuffer> _buf;

    ImageFileHandle       _file;
    SmartPtr<BufferPool>  _pool;
    SmartPtr<GeoMapper>   _mapper;
#if XCAM_TEST_OPENCV
    cv::VideoWriter       _writer;
#endif
};

typedef std::vector<SmartPtr<SoftElement>> SoftElements;

SoftElement::SoftElement (const char *file_name, uint32_t width, uint32_t height)
    : _file_name (NULL)
    , _width (width)
    , _height (height)
{
    if (file_name)
        _file_name = strndup (file_name, XCAM_TEST_MAX_STR_SIZE);
}

SoftElement::~SoftElement ()
{
    _file.close ();

    if (_file_name) {
        xcam_free (_file_name);
        _file_name = NULL;
    }
}

void
SoftElement::set_buf_size (uint32_t width, uint32_t height)
{
    _width = width;
    _height = height;
}

XCamReturn
SoftElement::open_file (const char *option)
{
    if (_file.open (_file_name, option) != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("open %s failed.", _file_name);
        return XCAM_RETURN_ERROR_FILE;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SoftElement::close_file ()
{
    return _file.close ();
}

XCamReturn
SoftElement::rewind_file ()
{
    return _file.rewind ();
}

XCamReturn
SoftElement::create_buf_pool (const VideoBufferInfo &info, uint32_t count)
{
    SmartPtr<BufferPool> pool = new SoftVideoBufAllocator ();
    XCAM_ASSERT (pool.ptr ());
    _pool = pool;

    _pool->set_video_info (info);
    if (!_pool->reserve (count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
SoftElement::read_buf ()
{
    _buf = _pool->get_buffer (_pool);
    XCAM_ASSERT (_buf.ptr ());

    return _file.read_buf (_buf);
}

XCamReturn
SoftElement::write_buf () {
    return _file.write_buf (_buf);
}

#if XCAM_TEST_OPENCV
XCamReturn
SoftElement::cv_open_writer ()
{
    XCAM_FAIL_RETURN (
        ERROR,
        _width && _height,
        XCAM_RETURN_ERROR_PARAM,
        "invalid size width:%d height:%d", _width, _height);

    cv::Size frame_size = cv::Size (_width, _height);
    if (!_writer.open (_file_name, CV_FOURCC('X', '2', '6', '4'), 30, frame_size)) {
        XCAM_LOG_ERROR ("open file %s failed", _file_name);
        return XCAM_RETURN_ERROR_FILE;
    }

    return XCAM_RETURN_NO_ERROR;
}

void
SoftElement::cv_write_image (char *img_name, char *frame_str, char *idx_str)
{
    cv::Mat mat;

#if XCAM_TEST_SOFT_IMAGE_DEBUG
    convert_to_mat (_buf, mat);

    cv::putText (mat, frame_str, cv::Point(20, 50), fontFace, 2.0, color, 2, 8, false);
    if(idx_str)
        cv::putText (mat, idx_str, cv::Point(20, 110), fontFace, 2.0, color, 2, 8, false);

    cv::imwrite (img_name, mat);
#else
    XCAM_UNUSED (img_name);
    XCAM_UNUSED (frame_str);
    XCAM_UNUSED (idx_str);
#endif

    if (_writer.isOpened ()) {
        if (mat.empty())
            convert_to_mat (_buf, mat);

        _writer.write (mat);
    }
}
#endif

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
    const char *dir_delimiter = std::strrchr (orig_name, '/');

    if (dir_delimiter) {
        std::string path (orig_name, dir_delimiter - orig_name + 1);
        XCAM_ASSERT (path.c_str ());
        snprintf (new_name, XCAM_TEST_MAX_STR_SIZE, "%s%s_%s", path.c_str (), embedded_str, dir_delimiter + 1);
    } else {
        snprintf (new_name, XCAM_TEST_MAX_STR_SIZE, "%s_%s", embedded_str, orig_name);
    }
}

static void
add_element (SoftElements &elements, const char *element_name, uint32_t width, uint32_t height)
{
    char file_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    combine_name (elements[0]->get_file_name (), element_name, file_name);

    SmartPtr<SoftElement> element = new SoftElement (file_name, width, height);
    elements.push_back (element);
}

static XCamReturn
elements_open_file (const SoftElements &elements, const char *option, const bool &nv12_output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    for (uint32_t i = 0; i < elements.size (); ++i) {
        if (nv12_output)
            ret = elements[i]->open_file (option);
#if XCAM_TEST_OPENCV
        else
            ret = elements[i]->cv_open_writer ();
#endif

        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("open file(%s) failed", elements[i]->get_file_name ());
            break;
        }
    }

    return ret;
}

static void
write_image (const SoftElements &ins, const SoftElements &outs, const bool &nv12_output) {
    if (nv12_output) {
        for (uint32_t i = 0; i < outs.size (); ++i)
            outs[i]->write_buf ();
    }
#if XCAM_TEST_OPENCV
    else {
        static uint32_t frame_num = 0;
        char img_name[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
        char frame_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
        std::snprintf (frame_str, XCAM_TEST_MAX_STR_SIZE, "frame:%d", frame_num);

        char idx_str[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
        for (uint32_t i = 0; i < ins.size (); ++i) {
            std::snprintf (idx_str, XCAM_TEST_MAX_STR_SIZE, "idx:%d", i);
            std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "orig_fisheye_%d_%d.jpg", frame_num, i);
            ins[i]->cv_write_image (img_name, frame_str, idx_str);
        }

        for (uint32_t i = 0; i < outs.size (); ++i) {
            std::snprintf (img_name, XCAM_TEST_MAX_STR_SIZE, "%s_%d.jpg", outs[i]->get_file_name (), frame_num);
            outs[i]->cv_write_image (img_name, frame_str);
        }
        frame_num++;
    }
#endif
}

static XCamReturn
ensure_output_format (const char *file_name, const SoftType &type, bool &nv12_output)
{
    char suffix[XCAM_TEST_MAX_STR_SIZE] = {'\0'};
    const char *ptr = std::strrchr (file_name, '.');
    std::snprintf (suffix, XCAM_TEST_MAX_STR_SIZE, "%s", ptr + 1);
    if (!strcasecmp (suffix, "mp4")) {
#if XCAM_TEST_OPENCV
        if (type != SoftTypeStitch) {
            XCAM_LOG_ERROR ("only stitch type supports MP4 output format");
            return XCAM_RETURN_ERROR_PARAM;
        }
        nv12_output = false;
#else
        XCAM_LOG_ERROR ("only supports NV12 output format");
        return XCAM_RETURN_ERROR_PARAM;
#endif
    }

    return XCAM_RETURN_NO_ERROR;
}

static bool
check_element (const SoftElements &elements, const uint32_t &idx)
{
    if (idx >= elements.size ())
        return false;

    if (!elements[idx].ptr()) {
        XCAM_LOG_ERROR ("SoftElement(idx:%d) ptr is NULL", idx);
        return false;
    }

    XCAM_FAIL_RETURN (
        ERROR,
        elements[idx]->get_width () && elements[idx]->get_height (),
        false,
        "SoftElement(idx:%d): invalid parameters width:%d height:%d",
        idx, elements[idx]->get_width (), elements[idx]->get_height ());

    return true;
}

static XCamReturn
check_elements (const SoftElements &elements)
{
    for (uint32_t i = 0; i < elements.size (); ++i) {
        XCAM_FAIL_RETURN (
            ERROR,
            check_element (elements, i),
            XCAM_RETURN_ERROR_PARAM,
            "invalid SoftElement index:%d\n", i);
    }

    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
create_topview_mapper (
    const SmartPtr<Stitcher> &stitcher,
    const SmartPtr<SoftElement> &stitch, const SmartPtr<SoftElement> &topview)
{
    BowlModel bowl_model (stitcher->get_bowl_config (), stitch->get_width (), stitch->get_height ());
    BowlModel::PointMap points;

    float length_mm = 0.0f, width_mm = 0.0f;
    bowl_model.get_max_topview_area_mm (length_mm, width_mm);
    XCAM_LOG_INFO ("Max Topview Area (L%.2fmm, W%.2fmm)", length_mm, width_mm);

    bowl_model.get_topview_rect_map (points, topview->get_width (), topview->get_height (), length_mm, width_mm);
    SmartPtr<GeoMapper> mapper = GeoMapper::create_soft_geo_mapper ();
    XCAM_ASSERT (mapper.ptr ());

    mapper->set_output_size (topview->get_width (), topview->get_height ());
    mapper->set_lookup_table (points.data (), topview->get_width (), topview->get_height ());
    topview->set_mapper (mapper);

    return XCAM_RETURN_NO_ERROR;
}

static XCamReturn
remap_topview_buf (const SmartPtr<SoftElement> &stitch, const SmartPtr<SoftElement> &topview)
{
    SmartPtr<GeoMapper> mapper = topview->get_mapper();
    XCAM_ASSERT (mapper.ptr ());

    XCamReturn ret = mapper->remap (stitch->get_buf (), topview->get_buf ());
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("remap stitched image to topview failed.");
        return ret;
    }

#if 0
    BowlModel::VertexMap bowl_vertices;
    BowlModel::PointMap bowl_points;
    uint32_t bowl_lut_w = 15, bowl_lut_h = 10;
    model.get_bowlview_vertex_map (bowl_vertices, bowl_points, bowl_lut_w, bowl_lut_h);
    for (uint32_t i = 0; i < bowl_lut_h; ++i) {
        for (uint32_t j = 0; j < bowl_lut_w; ++j)
        {
            PointFloat3 &vetex = bowl_vertices[i * bowl_lut_w + j];
            printf ("(%4.0f, %4.0f, %4.0f), ", vetex.x, vetex.y, vetex.z );
        }
        printf ("\n");
    }
#endif

    return XCAM_RETURN_NO_ERROR;
}

static int
run_stitcher (
    const SmartPtr<Stitcher> &stitcher,
    const SoftElements &ins, const SoftElements &outs,
    bool nv12_output, bool save_output, int loop)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    CHECK (check_elements (ins), "invalid input elements");
    CHECK (check_elements (outs), "invalid output elements");

    VideoBufferList in_buffers;
    while (loop--) {
        for (uint32_t i = 0; i < ins.size (); ++i) {
            CHECK (ins[i]->rewind_file (), "rewind buffer from file(%s) failed", ins[i]->get_file_name ());
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
                stitcher->stitch_buffers (in_buffers, outs[0]->get_buf ()),
                "stitch buffer failed.");

            if (save_output) {
                if (check_element (outs, 1)) {
                    CHECK (remap_topview_buf (outs[0], outs[1]), "run topview failed");
                }

                write_image (ins, outs, nv12_output);
            }

            FPS_CALCULATION (soft - stitcher, XCAM_OBJ_DUR_FRAME_NUM);
        } while (true);
    }

    return 0;
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --type TYPE--input0 file0 --input1 file1 --output file\n"
            "\t--type              processing type, selected from: blend, remap, stitch, ...\n"
            "\t--                  [stitch]: read calibration files from exported path $FISHEYE_CONFIG_PATH\n"
            "\t--input0            input image(NV12)\n"
            "\t--input1            input image(NV12)\n"
            "\t--input2            input image(NV12)\n"
            "\t--input3            input image(NV12)\n"
            "\t--output            output image(NV12)\n"
            "\t--in-w              optional, input width, default: 1920\n"
            "\t--in-h              optional, input height, default: 1080\n"
            "\t--out-w             optional, output width, default: 1920\n"
            "\t--out-h             optional, output height, default: 960\n"
            "\t--topview-w         optional, output width, default: 1280\n"
            "\t--topview-h         optional, output height, default: 720\n"
            "\t--scale-mode        optional, scaling mode for geometric mapping,\n"
            "\t                    select from [singleconst/dualconst], default: singleconst\n"
            "\t--save              optional, save file or not, select from [true/false], default: true\n"
            "\t--loop              optional, how many loops need to run, default: 1\n"
            "\t--help              usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t output_width = 1920; //output_height * 2;
    uint32_t output_height = 960; //960;
    uint32_t topview_width = 1280;
    uint32_t topview_height = 720;
    SoftType type = SoftTypeNone;
    GeoMapScaleMode scale_mode = ScaleSingleConst;

    SoftElements ins;
    SoftElements outs;

    int loop = 1;
    bool save_output = true;
    bool nv12_output = true;

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
        {"topview-w", required_argument, NULL, 'P'},
        {"topview-h", required_argument, NULL, 'V'},
        {"scale-mode", required_argument, NULL, 'S'},
        {"save", required_argument, NULL, 's'},
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
            ADD_ENELEMT(ins, optarg);
            break;
        case 'j':
            XCAM_ASSERT (optarg);
            ADD_ENELEMT(ins, optarg);
            break;
        case 'k':
            XCAM_ASSERT (optarg);
            ADD_ENELEMT(ins, optarg);
            break;
        case 'l':
            XCAM_ASSERT (optarg);
            ADD_ENELEMT(ins, optarg);
            break;
        case 'o':
            XCAM_ASSERT (optarg);
            ADD_ENELEMT(outs, optarg);
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
            else {
                XCAM_LOG_ERROR ("GeoMapScaleMode unknown mode: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 's':
            save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
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
    printf ("topview width:\t\t%d\n", topview_width);
    printf ("topview height:\t\t%d\n", topview_height);
    printf ("scaling mode:\t\t%s\n", (scale_mode == ScaleSingleConst) ? "singleconst" : "dualconst");
    printf ("save output:\t\t%s\n", save_output ? "true" : "false");
    printf ("loop count:\t\t%d\n", loop);

    VideoBufferInfo in_info, out_info;
    in_info.init (V4L2_PIX_FMT_NV12, input_width, input_height);
    out_info.init (V4L2_PIX_FMT_NV12, output_width, output_height);

    for (uint32_t i = 0; i < ins.size (); ++i) {
        ins[i]->set_buf_size (input_width, input_height);
        CHECK (ins[i]->create_buf_pool (in_info, 6), "create buffer pool failed");
        CHECK (ins[i]->open_file ("rb"), "open file(%s) failed", ins[i]->get_file_name ());
    }

    outs[0]->set_buf_size (output_width, output_height);
    if (save_output) {
        CHECK (ensure_output_format (outs[0]->get_file_name (), type, nv12_output), "unsupported output format");
        if (nv12_output) {
            CHECK (outs[0]->open_file ("wb"), "open file(%s) failed", outs[0]->get_file_name ());
        }
    }

    switch (type) {
    case SoftTypeBlender: {
        CHECK_EXP (ins.size () >= 2, "blender need 2 input files.");
        SmartPtr<Blender> blender = Blender::create_soft_blender ();
        XCAM_ASSERT (blender.ptr ());
        blender->set_output_size (output_width, output_height);
        Rect merge_window;
        merge_window.pos_x = 0;
        merge_window.pos_y = 0;
        merge_window.width = out_info.width;
        merge_window.height = out_info.height;
        blender->set_merge_window (merge_window);

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        CHECK (ins[1]->read_buf(), "read buffer from file(%s) failed.", ins[1]->get_file_name ());
        RUN_N (blender->blend (ins[0]->get_buf (), ins[1]->get_buf (), outs[0]->get_buf ()), loop, "blend buffer failed.");
        if (save_output)
            outs[0]->write_buf ();
        break;
    }
    case SoftTypeRemap: {
        SmartPtr<GeoMapper> mapper = GeoMapper::create_soft_geo_mapper ();
        XCAM_ASSERT (mapper.ptr ());
        mapper->set_output_size (output_width, output_height);
        mapper->set_lookup_table (map_table, MAP_WIDTH, MAP_HEIGHT);
        //mapper->set_factors ((output_width - 1.0f) / (MAP_WIDTH - 1.0f), (output_height - 1.0f) / (MAP_HEIGHT - 1.0f));

        CHECK (ins[0]->read_buf(), "read buffer from file(%s) failed.", ins[0]->get_file_name ());
        RUN_N (mapper->remap (ins[0]->get_buf (), outs[0]->get_buf ()), loop, "remap buffer failed.");
        if (save_output)
            outs[0]->write_buf ();
        break;
    }
    case SoftTypeStitch: {
        CHECK_EXP (ins.size () >= 2 && ins.size () <= 4, "stitcher need at 2~4 input files.");

        uint32_t camera_count = ins.size ();
        SmartPtr<Stitcher> stitcher = Stitcher::create_soft_stitcher ();
        XCAM_ASSERT (stitcher.ptr ());

        CameraInfo cam_info[4];
        const char *fisheye_config_path = getenv (FISHEYE_CONFIG_ENV_VAR);
        if (!fisheye_config_path)
            fisheye_config_path = FISHEYE_CONFIG_PATH;

        XCAM_LOG_INFO ("calibration config path:%s", XCAM_STR (fisheye_config_path));

        for (uint32_t i = 0; i < camera_count; ++i) {
            if (parse_camera_info (fisheye_config_path, i, cam_info[i], camera_count) != 0) {
                XCAM_LOG_ERROR ("parse fisheye dewarp info(idx:%d) failed.", i);
                return -1;
            }
        }

        PointFloat3 bowl_coord_offset;
        if (camera_count == 4) {
            centralize_bowl_coord_from_cameras (
                cam_info[0].calibration.extrinsic, cam_info[1].calibration.extrinsic,
                cam_info[2].calibration.extrinsic, cam_info[3].calibration.extrinsic,
                bowl_coord_offset);
        }

        stitcher->set_camera_num (camera_count);
        for (uint32_t i = 0; i < camera_count; ++i) {
            stitcher->set_camera_info (i, cam_info[i]);
        }

        BowlDataConfig bowl;
        bowl.wall_height = 3000.0f;
        bowl.ground_length = 2000.0f;
        //bowl.a = 5000.0f;
        //bowl.b = 3600.0f;
        //bowl.c = 3000.0f;
        bowl.angle_start = 0.0f;
        bowl.angle_end = 360.0f;
        stitcher->set_bowl_config (bowl);
        stitcher->set_output_size (output_width, output_height);
        stitcher->set_scale_mode (scale_mode);

        if (save_output) {
            add_element (outs, "topview", topview_width, topview_height);
            elements_open_file (outs, "wb", nv12_output);

            create_topview_mapper (stitcher, outs[0], outs[1]);
        }
        CHECK_EXP (
            run_stitcher (stitcher, ins, outs, nv12_output, save_output, loop) == 0,
            "run stitcher failed.");
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
