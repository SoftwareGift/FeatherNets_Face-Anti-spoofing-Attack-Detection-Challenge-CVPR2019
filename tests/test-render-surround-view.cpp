/*
 * test-render-surround-view.cpp - test render surround view
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
 * Author: Zong Wei <wei.zong@intel.com>
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
#if HAVE_VULKAN
#include <vulkan/vk_device.h>
#endif

#include <render/render_osg_viewer.h>
#include <render/render_osg_model.h>
#include <render/render_osg_shader.h>

using namespace XCam;

enum SVModule {
    SVModuleNone    = 0,
    SVModuleSoft,
    SVModuleGLES,
    SVModuleVulkan
};

#define CAR_MODEL_NAME  "Suv.osgb"

static const char VtxShaderCar[] = ""
                                   "precision highp float;                                        \n"
                                   "uniform mat4 osg_ModelViewProjectionMatrix;                   \n"
                                   "uniform mat4 osg_ModelViewMatrix;                             \n"
                                   "uniform mat3 osg_NormalMatrix;                                \n"
                                   "attribute vec3 osg_Normal;                                    \n"
                                   "attribute vec4 osg_Color;                                     \n"
                                   "attribute vec4 osg_Vertex;                                    \n"
                                   "varying vec4 v_color;                                         \n"
                                   "varying float diffuseLight;                                   \n"
                                   "varying float specLight;                                      \n"
                                   "attribute vec2 osg_MultiTexCoord0;                            \n"
                                   "varying vec2 texCoord0;                                       \n"
                                   "void main()                                                   \n"
                                   "{                                                             \n"
                                   "    vec4 light = vec4(0.0,100.0, 100.0, 1.0);                 \n"
                                   "    vec4 lightColorSpec = vec4(1.0, 1.0, 1.0, 1.0);           \n"
                                   "    vec4 lightColorDiffuse = vec4(1.0, 1.0, 1.0, 1.0);        \n"
                                   "    vec4 lightColorAmbient = vec4(0.3, 0.3, .3, 1.0);         \n"
                                   "    vec4 carColorAmbient = vec4(0.0, 0.0, 1.0, 1.0);          \n"
                                   "    vec4 carColorDiffuse = vec4(0.0, 0.0, 1.0, 1.0);          \n"
                                   "    vec4 carColorSpec = vec4(1.0, 1.0, 1.0, 1.0);             \n"
                                   "    vec3 tnorm = normalize(osg_NormalMatrix * osg_Normal);    \n"
                                   "    vec4 eye = osg_ModelViewMatrix * osg_Vertex;              \n"
                                   "    vec3 s = normalize(vec3(light - eye));                    \n"
                                   "    vec3 v = normalize(-eye.xyz);                             \n"
                                   "    vec3 r = reflect(-s, tnorm);                              \n"
                                   "    diffuseLight = max(0.0, dot( s, tnorm));                  \n"
                                   "    specLight = 0.0;                                          \n"
                                   "    if(diffuseLight > 0.0)                                    \n"
                                   "    {                                                         \n"
                                   "        specLight = pow(max(0.0, dot(r,v)), 10.0);            \n"
                                   "    }                                                         \n"
                                   "    texCoord0 = osg_MultiTexCoord0;                               \n"
                                   "    v_color = (specLight *  lightColorSpec * carColorSpec) + (carColorDiffuse * lightColorDiffuse * diffuseLight) + lightColorAmbient * carColorAmbient;   \n"
                                   "    gl_Position = osg_ModelViewProjectionMatrix * osg_Vertex;     \n"
                                   "}                                                \n";

static const char FrgShaderCar[] = ""
                                   "precision highp float;                                                      \n"
                                   "varying vec4 v_color;                                                       \n"
                                   "varying float diffuseLight;                                                 \n"
                                   "varying float specLight;                                                    \n"
                                   "uniform sampler2D textureWheel;                                             \n"
                                   "varying vec2 texCoord0;                                                     \n"
                                   "void main()                                                                 \n"
                                   "{                                                                           \n"
                                   "    vec4 lightColorSpec = vec4(1.0, 1.0, 1.0, 1.0);                         \n"
                                   "    vec4 lightColorDiffuse = vec4(1.0, 1.0, 1.0, 1.0);                      \n"
                                   "    vec4 lightColorAmbient = vec4(0.3, 0.3, .3, 1.0);                       \n"
                                   "    vec4 base = texture2D(textureWheel, texCoord0.st);                      \n"
                                   "    gl_FragColor = (specLight *  lightColorSpec * base) + (base * lightColorDiffuse * diffuseLight) + lightColorAmbient * base ; \n"
                                   "}                                                                           \n";

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

#if HAVE_VULKAN
    void set_vk_device (SmartPtr<VKDevice> &device) {
        XCAM_ASSERT (device.ptr ());
        _vk_dev = device;
    }
    SmartPtr<VKDevice> &get_vk_device () {
        return _vk_dev;
    }
#endif

    virtual XCamReturn create_buf_pool (const VideoBufferInfo &info, uint32_t count);

private:
    XCAM_DEAD_COPY (SVStream);

private:
    SVModule               _module;
#if HAVE_VULKAN
    SmartPtr<VKDevice>     _vk_dev;
#endif
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
    } else if (_module == SVModuleVulkan) {
#if HAVE_VULKAN
        XCAM_ASSERT (_vk_dev.ptr ());
        pool = create_vk_buffer_pool (_vk_dev);
        XCAM_ASSERT (pool.ptr ());
        pool->set_video_info (info);
#endif
    }
    XCAM_ASSERT (pool.ptr ());

    if (!pool->reserve (count)) {
        XCAM_LOG_ERROR ("create buffer pool failed");
        pool.release ();
        return XCAM_RETURN_ERROR_MEM;
    }

    set_buf_pool (pool);
    return XCAM_RETURN_NO_ERROR;
}

static SmartPtr<Stitcher>
create_stitcher (const SmartPtr<SVStream> &stitch, SVModule module)
{
    SmartPtr<Stitcher> stitcher;

    if (module == SVModuleSoft) {
        stitcher = Stitcher::create_soft_stitcher ();
    } else if (module == SVModuleGLES) {
#if HAVE_GLES
        stitcher = Stitcher::create_gl_stitcher ();
#endif
    } else if (module == SVModuleVulkan) {
#if HAVE_VULKAN
        SmartPtr<VKDevice> dev = stitch->get_vk_device ();
        XCAM_ASSERT (dev.ptr ());
        stitcher = Stitcher::create_vk_stitcher (dev);
#else
        XCAM_UNUSED (stitch);
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

void
get_bowl_model (
    const SmartPtr<Stitcher> &stitcher,
    BowlModel::VertexMap &vertices,
    BowlModel::PointMap &points,
    BowlModel::IndexVector &indices,
    float &a,
    float &b,
    float &c,
    float resRatio,
    uint32_t image_width,
    uint32_t image_height)
{
    uint32_t res_width = image_width * resRatio;
    uint32_t res_height = image_height * resRatio;

    BowlDataConfig bowl = stitcher->get_bowl_config();
    bowl.angle_start = 0.0f;
    bowl.angle_end = 360.0f;

    a = bowl.a;
    b = bowl.b;
    c = bowl.c;

    BowlModel bowl_model(bowl, image_width, image_height);

    bowl_model.get_bowlview_vertex_model(
        vertices,
        points,
        indices,
        res_width,
        res_height);
}

static SmartPtr<RenderOsgModel>
create_surround_view_model (
    const SmartPtr<Stitcher> &stitcher,
    uint32_t texture_width,
    uint32_t texture_height)
{
    SmartPtr<RenderOsgModel> svm_model = new RenderOsgModel ("svm model", texture_width, texture_height);

    svm_model->setup_shader_program ("SVM", osg::Shader::VERTEX, VtxShaderProjectNV12Texture);
    svm_model->setup_shader_program ("SVM", osg::Shader::FRAGMENT, FrgShaderProjectNV12Texture);

    BowlModel::VertexMap vertices;
    BowlModel::PointMap points;
    BowlModel::IndexVector indices;

    float a = 0;
    float b = 0;
    float c = 0;
    float res_ratio = 0.3;
    float scaling = 1000.0f;

    get_bowl_model (stitcher, vertices, points, indices,
                    a, b, c, res_ratio, texture_width, texture_height );

    svm_model->setup_vertex_model (vertices, points, indices, a / scaling, b / scaling, c / scaling);

    return svm_model;
}

static SmartPtr<RenderOsgModel>
create_car_model (const char *name)
{
    std::string car_name;
    if (NULL != name) {
        car_name = std::string (name);
    } else {
        car_name = std::string (CAR_MODEL_NAME);
    }
    std::string car_model_path = FISHEYE_CONFIG_PATH + car_name;

    const char *env_path = std::getenv (FISHEYE_CONFIG_ENV_VAR);
    if (env_path) {
        car_model_path.clear ();
        car_model_path = std::string (env_path) + car_name;
    }

    SmartPtr<RenderOsgModel> car_model = new RenderOsgModel (car_model_path.c_str(), true);

    car_model->setup_shader_program ("Car", osg::Shader::VERTEX, VtxShaderCar);
    car_model->setup_shader_program ("Car", osg::Shader::FRAGMENT, FrgShaderCar);

    float translation_x = -0.3f;
    float translation_y = 0.0f;
    float translation_z = 0.0f;
    float rotation_x = 0.0f;
    float rotation_y = 0.0f;
    float rotation_z = 1.0f;
    float rotation_degrees = -180.0;

    car_model->setup_model_matrix (
        translation_x,
        translation_y,
        translation_z,
        rotation_x,
        rotation_y,
        rotation_z,
        rotation_degrees);

    return car_model;
}

static int
run_stitcher (
    const SmartPtr<Stitcher> &stitcher,
    const SmartPtr<RenderOsgModel> &model,
    const SVStreams &ins,
    const SVStreams &outs)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    Mutex mutex;

    VideoBufferList in_buffers;
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
        if (ret == XCAM_RETURN_BYPASS) {
            XCAM_LOG_DEBUG ("XCAM_RETURN_BYPASS \n");
            break;
        }

        {
            SmartLock locker (mutex);
            CHECK (
                stitcher->stitch_buffers (in_buffers, outs[0]->get_buf ()),
                "stitch buffer failed.");
        }

        model->update_texture (outs[0]->get_buf ());

        FPS_CALCULATION (render surround view, XCAM_OBJ_DUR_FRAME_NUM);
    } while (true);

    return 0;
}

static void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --module MODULE --input0 input.nv12 --input1 input1.nv12 --input2 input2.nv12 ...\n"
            "\t--module            processing module, selected from: soft, gles, vulkan\n"
            "\t--                  read calibration files from exported path $FISHEYE_CONFIG_PATH\n"
            "\t--input0            input image(NV12)\n"
            "\t--input1            input image(NV12)\n"
            "\t--input2            input image(NV12)\n"
            "\t--input3            input image(NV12)\n"
            "\t--in-w              optional, input width, default: 1280\n"
            "\t--in-h              optional, input height, default: 800\n"
            "\t--out-w             optional, output width, default: 1920\n"
            "\t--out-h             optional, output height, default: 640\n"
            "\t--scale-mode        optional, scaling mode for geometric mapping,\n"
            "\t                    select from [singleconst/dualconst/dualcurve], default: singleconst\n"
            "\t--fm-mode           optional, feature match mode,\n"
#if HAVE_OPENCV
            "\t                    select from [none/default/cluster/capi], default: none\n"
#else
            "\t                    select from [none], default: none\n"
#endif
            "\t--car               optional, car model name\n"
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

    SVStreams ins;
    SVStreams outs;
    PUSH_STREAM (SVStream, outs, NULL);

    const char *car_name = NULL;

    SVModule module = SVModuleGLES;
    GeoMapScaleMode scale_mode = ScaleSingleConst;
    FeatureMatchMode fm_mode = FMNone;

    int loop = 1;

    const struct option long_opts[] = {
        {"module", required_argument, NULL, 'm'},
        {"input0", required_argument, NULL, 'i'},
        {"input1", required_argument, NULL, 'j'},
        {"input2", required_argument, NULL, 'k'},
        {"input3", required_argument, NULL, 'l'},
        {"in-w", required_argument, NULL, 'w'},
        {"in-h", required_argument, NULL, 'h'},
        {"out-w", required_argument, NULL, 'W'},
        {"out-h", required_argument, NULL, 'H'},
        {"scale-mode", required_argument, NULL, 'S'},
        {"fm-mode", required_argument, NULL, 'F'},
        {"car", required_argument, NULL, 'c'},
        {"loop", required_argument, NULL, 'L'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'm':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "soft")) {
                module = SVModuleSoft;
            } else if (!strcasecmp (optarg, "gles")) {
                module = SVModuleGLES;
            } else if (!strcasecmp (optarg, "vulkan")) {
                module = SVModuleVulkan;
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
        case 'w':
            input_width = (uint32_t)atoi(optarg);
            break;
        case 'h':
            input_height = (uint32_t)atoi(optarg);
            break;
        case 'W':
            output_width = (uint32_t)atoi(optarg);
            break;
        case 'H':
            output_height = (uint32_t)atoi(optarg);
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
        case 'F':
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "none"))
                fm_mode = FMNone;
#if HAVE_OPENCV
            else if (!strcasecmp (optarg, "default"))
                fm_mode = FMDefault;
            else if (!strcasecmp (optarg, "cluster"))
                fm_mode = FMCluster;
            else if (!strcasecmp (optarg, "capi"))
                fm_mode = FMCapi;
#endif
            else {
                XCAM_LOG_ERROR ("unsupported feature match mode: %s", optarg);
                usage (argv[0]);
                return -1;
            }
            break;
        case 'c':
            XCAM_ASSERT (optarg);
            car_name = optarg;
            break;
        case 'L':
            loop = atoi(optarg);
            break;
        case 'e':
            usage (argv[0]);
            return 0;
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

    CHECK_EXP (outs.size () == 1 && outs[0].ptr (), "surrond view needs 1 output stream");

    for (uint32_t i = 0; i < ins.size (); ++i) {
        printf ("input%d file:\t\t%s\n", i, ins[i]->get_file_name ());
    }
    printf ("input width:\t\t%d\n", input_width);
    printf ("input height:\t\t%d\n", input_height);
    printf ("output width:\t\t%d\n", output_width);
    printf ("output height:\t\t%d\n", output_height);
    printf ("scaling mode:\t\t%s\n", (scale_mode == ScaleSingleConst) ? "singleconst" :
            ((scale_mode == ScaleDualConst) ? "dualconst" : "dualcurve"));
    printf ("feature match:\t\t%s\n", (fm_mode == FMNone) ? "none" :
            ((fm_mode == FMDefault ) ? "default" : ((fm_mode == FMCluster) ? "cluster" : "capi")));
    printf ("car model name:\t\t%s\n", car_name != NULL ? car_name : "Not specified, use default model");
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
#else
    if (module == SVModuleGLES) {
        XCAM_LOG_ERROR ("GLES module unsupported");
        return -1;
    }
#endif

#if HAVE_VULKAN
    if (module == SVModuleVulkan) {
        scale_mode = ScaleSingleConst;
        if (scale_mode != ScaleSingleConst) {
            XCAM_LOG_ERROR ("vulkan module only support singleconst scale mode currently");
            return -1;
        }

        SmartPtr<VKDevice> vk_dev = VKDevice::default_device ();
        for (uint32_t i = 0; i < ins.size (); ++i) {
            ins[i]->set_vk_device (vk_dev);
        }
        XCAM_ASSERT (outs[0].ptr ());
        outs[0]->set_vk_device (vk_dev);
    }
#else
    if (module == SVModuleVulkan) {
        XCAM_LOG_ERROR ("vulkan module unsupported");
        return -1;
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

    outs[0]->set_buf_size (output_width, output_height);

    SmartPtr<Stitcher> stitcher = create_stitcher (outs[0], module);
    XCAM_ASSERT (stitcher.ptr ());

    CameraInfo cam_info[4];
    std::string fisheye_config_path = FISHEYE_CONFIG_PATH;
    const char *env = std::getenv (FISHEYE_CONFIG_ENV_VAR);
    if (env)
        fisheye_config_path.assign (env, strlen (env));
    XCAM_LOG_INFO ("calibration config path:%s", fisheye_config_path.c_str ());

    uint32_t camera_count = ins.size ();
    for (uint32_t i = 0; i < camera_count; ++i) {
        if (parse_camera_info (fisheye_config_path.c_str (), i, cam_info[i], camera_count) != 0) {
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
    stitcher->set_fm_mode (fm_mode);

    SmartPtr<RenderOsgViewer> render = new RenderOsgViewer ();

    SmartPtr<RenderOsgModel> sv_model = create_surround_view_model (stitcher, output_width, output_height);
    render->add_model (sv_model);

    SmartPtr<RenderOsgModel> car_model = create_car_model (car_name);
    render->add_model (car_model);

    render->validate_model_groups ();

    render->start_render ();

    while (loop--) {
        CHECK_EXP (
            run_stitcher (stitcher, sv_model, ins, outs) == 0,
            "run stitcher failed");
    }

    render->stop_render ();

    return 0;
}
