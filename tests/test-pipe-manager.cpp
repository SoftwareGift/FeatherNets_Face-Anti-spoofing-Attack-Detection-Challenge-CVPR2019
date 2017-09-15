/*
  * test-pipe-manager.cpp -test pipe manager
  *
  *  Copyright (c) 2016 Intel Corporation
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
  * Author: Yinhang Liu <yinhangx.liu@intel.com>
  */

#include "pipe_manager.h"
#include "smart_analyzer_loader.h"
#include "ocl/cl_post_image_processor.h"
#if HAVE_LIBDRM
#include "drm_display.h"
#endif
#include <getopt.h>
#include "test_common.h"
#include <signal.h>
#include <stdio.h>

#define DEFAULT_FPT_BUF_COUNT 32

using namespace XCam;

static bool is_stop = false;

struct FileFP {
    FILE *fp;
    FileFP ()
        : fp (NULL)
    {}
    ~FileFP ()
    {
        if (fp)
            fclose (fp);
        fp = NULL;
    }
};

class MainPipeManager
    : public PipeManager
{
public:
    MainPipeManager ()
        : _image_width (0)
        , _image_height (0)
        , _enable_display (false)
    {
#if HAVE_LIBDRM
        _display = DrmDisplay::instance ();
#endif
        XCAM_OBJ_PROFILING_INIT;
    }

    void set_image_width (uint32_t image_width) {
        _image_width = image_width;
    }

    void set_image_height (uint32_t image_height) {
        _image_height = image_height;
    }

    void enable_display (bool value) {
        _enable_display = value;
    }

    void set_display_mode (DrmDisplayMode mode) {
        _display->set_display_mode (mode);
    }

protected:
    virtual void post_buffer (const SmartPtr<VideoBuffer> &buf);
    int display_buf (const SmartPtr<VideoBuffer> &buf);

private:
    uint32_t              _image_width;
    uint32_t              _image_height;
    bool                  _enable_display;
    SmartPtr<DrmDisplay>  _display;
    XCAM_OBJ_PROFILING_DEFINES;
};

void
MainPipeManager::post_buffer (const SmartPtr<VideoBuffer> &buf)
{
    FPS_CALCULATION (fps_buf, XCAM_OBJ_DUR_FRAME_NUM);

    XCAM_OBJ_PROFILING_START;

    if (_enable_display)
        display_buf (buf);

    XCAM_OBJ_PROFILING_END("main_pipe_manager_display", XCAM_OBJ_DUR_FRAME_NUM);
}

int
MainPipeManager::display_buf (const SmartPtr<VideoBuffer> &data)
{
#if HAVE_LIBDRM
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<VideoBuffer> buf = data;
    const VideoBufferInfo & frame_info = buf->get_video_info ();
    struct v4l2_rect rect = { 0, 0, frame_info.width, frame_info.height};

    if (!_display->is_render_inited ()) {
        ret = _display->render_init (0, 0, this->_image_width, this->_image_height,
                                     frame_info.format, &rect);
        CHECK (ret, "display failed on render_init");
    }
    ret = _display->render_setup_frame_buffer (buf);
    CHECK (ret, "display failed on framebuf set");
    ret = _display->render_buffer (buf);
    CHECK (ret, "display failed on rendering");
#endif
    return 0;
}

XCamReturn
read_buf (SmartPtr<VideoBuffer> &buf, FileFP &file)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    uint8_t *memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fread (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, file.fp) != line_bytes) {
                if (feof (file.fp)) {
                    fseek (file.fp, 0, SEEK_SET);
                    ret = XCAM_RETURN_BYPASS;
                } else {
                    XCAM_LOG_ERROR ("read file failed, size doesn't match");
                    ret = XCAM_RETURN_ERROR_FILE;
                }
                goto done;
            }
        }
    }
done:
    buf->unmap ();
    return ret;
}

void pipe_stop_handler(int sig)
{
    XCAM_UNUSED (sig);
    is_stop = true;
}

void print_help (const char *bin_name)
{
    printf ("Usage: %s [--format=NV12] [--width=1920] ...\n"
            "\t --format           specify output pixel format, default is NV12\n"
            "\t --width            specify input image width, default is 1920\n"
            "\t --height           specify input image height, default is 1080\n"
            "\t --fake-input       specify the path of image as fake source\n"
            "\t --defog-mode       specify defog mode\n"
            "\t                    select from [disabled, retinex, dcp], default is [disabled]\n"
            "\t --wavelet-mode     specify wavelet denoise mode, default is disable\n"
            "\t                    select from [0:disable, 1:Hat Y, 2:Hat UV, 3:Haar Y, 4:Haar UV, 5:Haar YUV, 6:Haar Bayes Shrink]\n"
            "\t --3d-denoise       specify 3D Denoise mode\n"
            "\t                    select from [disabled, yuv, uv], default is [disabled]\n"
            "\t --enable-wireframe enable wire frame\n"
            "\t --enable-warp      enable image warp\n"
            "\t --display-mode     display mode\n"
            "\t                    select from [primary, overlay], default is [primary]\n"
            "\t -p                 enable local display, need root privilege\n"
            "\t -h                 help\n"
            , bin_name);
}

int main (int argc, char *argv[])
{
    const char *bin_name = argv[0];

    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    VideoBufferInfo buf_info;
    SmartPtr<VideoBuffer> video_buf;
    SmartPtr<SmartAnalyzer> smart_analyzer;
    SmartPtr<CLPostImageProcessor> cl_post_processor;
    SmartPtr<BufferPool> drm_buf_pool;

    uint32_t pixel_format = V4L2_PIX_FMT_NV12;
    uint32_t image_width = 1920;
    uint32_t image_height = 1080;
    bool need_display = false;
    DrmDisplayMode display_mode = DRM_DISPLAY_MODE_PRIMARY;
    const char *input_path = NULL;
    FileFP input_fp;

    uint32_t defog_mode = 0;
    CLWaveletBasis wavelet_mode = CL_WAVELET_DISABLED;
    uint32_t wavelet_channel = CL_IMAGE_CHANNEL_UV;
    bool wavelet_bayes_shrink = false;
    uint32_t denoise_3d_mode = 0;
    uint8_t denoise_3d_ref_count = 3;
    bool enable_wireframe = false;
    bool enable_image_warp = false;

    int opt;
    const char *short_opts = "ph";
    const struct option long_opts [] = {
        {"format", required_argument, NULL, 'F'},
        {"width", required_argument, NULL, 'W'},
        {"height", required_argument, NULL, 'H'},
        {"fake-input", required_argument, NULL, 'A'},
        {"defog-mode", required_argument, NULL, 'D'},
        {"wavelet-mode", required_argument, NULL, 'V'},
        {"3d-denoise", required_argument, NULL, 'N'},
        {"enable-wireframe", no_argument, NULL, 'I'},
        {"enable-warp", no_argument, NULL, 'S'},
        {"display-mode", required_argument, NULL, 'P'},
        {NULL, 0, NULL, 0}
    };

    while ((opt = getopt_long (argc, argv, short_opts, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'F': {
            XCAM_ASSERT (optarg);
            CHECK_EXP ((strlen (optarg) == 4), "invalid pixel format\n");
            pixel_format = v4l2_fourcc ((unsigned) optarg[0],
                                        (unsigned) optarg[1],
                                        (unsigned) optarg[2],
                                        (unsigned) optarg[3]);
            break;
        }
        case 'W': {
            XCAM_ASSERT (optarg);
            image_width = atoi (optarg);
            break;
        }
        case 'H': {
            XCAM_ASSERT (optarg);
            image_height = atoi (optarg);
            break;
        }
        case 'A': {
            XCAM_ASSERT (optarg);
            XCAM_LOG_INFO ("use image %s as input source", optarg);
            input_path = optarg;
            break;
        }
        case 'D': {
            XCAM_ASSERT (optarg);
            if (!strcmp (optarg, "disabled"))
                defog_mode = CLPostImageProcessor::DefogDisabled;
            else if (!strcmp (optarg, "retinex"))
                defog_mode = CLPostImageProcessor::DefogRetinex;
            else if (!strcmp (optarg, "dcp"))
                defog_mode = CLPostImageProcessor::DefogDarkChannelPrior;
            else {
                print_help (bin_name);
                return -1;
            }
            break;
        }
        case 'V': {
            XCAM_ASSERT (optarg);
            if (atoi(optarg) < 0 || atoi(optarg) > 255) {
                print_help (bin_name);
                return -1;
            }
            if (atoi(optarg) == 1) {
                wavelet_mode = CL_WAVELET_HAT;
                wavelet_channel = CL_IMAGE_CHANNEL_Y;
            } else if (atoi(optarg) == 2) {
                wavelet_mode = CL_WAVELET_HAT;
                wavelet_channel = CL_IMAGE_CHANNEL_UV;
            } else if (atoi(optarg) == 3) {
                wavelet_mode = CL_WAVELET_HAAR;
                wavelet_channel = CL_IMAGE_CHANNEL_Y;
            } else if (atoi(optarg) == 4) {
                wavelet_mode = CL_WAVELET_HAAR;
                wavelet_channel = CL_IMAGE_CHANNEL_UV;
            } else if (atoi(optarg) == 5) {
                wavelet_mode = CL_WAVELET_HAAR;
                wavelet_channel = CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y;
            } else if (atoi(optarg) == 6) {
                wavelet_mode = CL_WAVELET_HAAR;
                wavelet_channel = CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y;
                wavelet_bayes_shrink = true;
            } else {
                wavelet_mode = CL_WAVELET_DISABLED;
            }
            break;
        }
        case 'N': {
            XCAM_ASSERT (optarg);
            if (!strcmp (optarg, "disabled"))
                denoise_3d_mode = CLPostImageProcessor::Denoise3DDisabled;
            else if (!strcmp (optarg, "yuv"))
                denoise_3d_mode = CLPostImageProcessor::Denoise3DYuv;
            else if (!strcmp (optarg, "uv"))
                denoise_3d_mode = CLPostImageProcessor::Denoise3DUV;
            else {
                print_help (bin_name);
                return -1;
            }
            break;
        }
        case 'I': {
            enable_wireframe = true;
            break;
        }
        case 'S': {
            enable_image_warp = true;
            break;
        }
        case 'P': {
            XCAM_ASSERT (optarg);
            if (!strcmp (optarg, "primary"))
                display_mode = DRM_DISPLAY_MODE_PRIMARY;
            else if (!strcmp (optarg, "overlay"))
                display_mode = DRM_DISPLAY_MODE_OVERLAY;
            else {
                print_help (bin_name);
                return -1;
            }
            break;
        }
        case 'p': {
            need_display = true;
            break;
        }
        case 'h':
            print_help (bin_name);
            return 0;
        default:
            print_help (bin_name);
            return -1;
        }
    }

    signal (SIGINT, pipe_stop_handler);

    if (!input_path) {
        XCAM_LOG_ERROR ("path of image is NULL");
        return -1;
    }
    input_fp.fp = fopen (input_path, "rb");
    if (!input_fp.fp) {
        XCAM_LOG_ERROR ("failed to open file: %s", XCAM_STR (input_path));
        return -1;
    }

    if (need_display && !DrmDisplay::set_preview (need_display)) {
        need_display = false;
        XCAM_LOG_WARNING ("set preview failed, disable local preview now");
    }

    SmartPtr<MainPipeManager> pipe_manager = new MainPipeManager ();
    pipe_manager->set_image_width (image_width);
    pipe_manager->set_image_height (image_height);
    pipe_manager->enable_display (need_display);
    pipe_manager->set_display_mode (display_mode);

    SmartHandlerList smart_handlers = SmartAnalyzerLoader::load_smart_handlers (DEFAULT_SMART_ANALYSIS_LIB_DIR);
    if (!smart_handlers.empty () ) {
        smart_analyzer = new SmartAnalyzer ();
        if (smart_analyzer.ptr ()) {
            SmartHandlerList::iterator i_handler = smart_handlers.begin ();
            for (; i_handler != smart_handlers.end (); ++i_handler) {
                XCAM_ASSERT ((*i_handler).ptr ());
                smart_analyzer->add_handler (*i_handler);
            }
        } else {
            XCAM_LOG_INFO ("load smart analyzer(%s) failed", DEFAULT_SMART_ANALYSIS_LIB_DIR);
        }
    }
    if (smart_analyzer.ptr ()) {
        if (smart_analyzer->prepare_handlers () != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("analyzer(%s) prepare handlers failed", smart_analyzer->get_name ());
        }
        pipe_manager->set_smart_analyzer (smart_analyzer);
    }

    cl_post_processor = new CLPostImageProcessor ();
    cl_post_processor->set_stats_callback (pipe_manager);
    cl_post_processor->set_defog_mode ((CLPostImageProcessor::CLDefogMode) defog_mode);
    cl_post_processor->set_wavelet (wavelet_mode, wavelet_channel, wavelet_bayes_shrink);
    cl_post_processor->set_3ddenoise_mode ((CLPostImageProcessor::CL3DDenoiseMode) denoise_3d_mode, denoise_3d_ref_count);

    cl_post_processor->set_wireframe (enable_wireframe);
    cl_post_processor->set_image_warp (enable_image_warp);
    if (smart_analyzer.ptr () && (enable_wireframe || enable_image_warp)) {
        cl_post_processor->set_scaler (true);
        cl_post_processor->set_scaler_factor (640.0 / image_width);
    }

    if (need_display) {
        cl_post_processor->set_output_format (V4L2_PIX_FMT_XBGR32);
    }
    pipe_manager->add_image_processor (cl_post_processor);

    ret = pipe_manager->start ();
    CHECK (ret, "pipe manager start failed");

    buf_info.init (pixel_format, image_width, image_height);
    SmartPtr<DrmDisplay> display = DrmDisplay::instance ();
    drm_buf_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (drm_buf_pool.ptr ());
    if (!drm_buf_pool->set_video_info (buf_info) || !drm_buf_pool->reserve (DEFAULT_FPT_BUF_COUNT)) {
        XCAM_LOG_ERROR ("init drm buffer pool failed");
        return -1;
    }

    while (!is_stop) {
        video_buf = drm_buf_pool->get_buffer (drm_buf_pool);
        XCAM_ASSERT (video_buf.ptr ());

        ret = read_buf (video_buf, input_fp);
        if (ret == XCAM_RETURN_BYPASS) {
            ret = read_buf (video_buf, input_fp);
        }

        if (ret == XCAM_RETURN_NO_ERROR)
            pipe_manager->push_buffer (video_buf);
    }

    ret = pipe_manager->stop();
    CHECK (ret, "pipe manager stop failed");

    return 0;
}
