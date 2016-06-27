/*
 * main.cpp - test
 *
 *  Copyright (c) 2014 Intel Corporation
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

#include "device_manager.h"
#include "atomisp_device.h"
#include "uvc_device.h"
#include "fake_v4l2_device.h"
#include "isp_controller.h"
#include "isp_image_processor.h"
#include "x3a_analyzer_simple.h"
#include "analyzer_loader.h"
#include "smart_analyzer_loader.h"
#if HAVE_IA_AIQ
#include "x3a_analyzer_aiq.h"
#include "x3a_analyze_tuner.h"
#endif
#if HAVE_LIBCL
#include "cl_3a_image_processor.h"
#include "cl_post_image_processor.h"
#include "cl_csc_image_processor.h"
#include "cl_hdr_handler.h"
#include "cl_tnr_handler.h"
#endif
#if HAVE_LIBDRM
#include "drm_display.h"
#endif
#include "dynamic_analyzer_loader.h"
#include "hybrid_analyzer_loader.h"
#include "isp_poll_thread.h"
#include "fake_poll_thread.h"
#include <base/xcam_3a_types.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>
#include "test_common.h"

using namespace XCam;

#define IMX185_WDR_CPF "/etc/atomisp/imx185_wdr.cpf"

static Mutex g_mutex;
static Cond  g_cond;
static bool  g_stop = false;

class MainDeviceManager
    : public DeviceManager
{
public:
    MainDeviceManager ()
        : _file (NULL)
        , _save_file (false)
        , _interval (1)
        , _frame_width (0)
        , _frame_height (0)
        , _frame_count (0)
        , _frame_save (0)
        , _enable_display (false)
    {
#if HAVE_LIBDRM
        _display = DrmDisplay::instance();
#endif
        XCAM_OBJ_PROFILING_INIT;
    }

    ~MainDeviceManager () {
        close_file ();
    }

    void enable_save_file (bool enable) {
        _save_file = enable;
    }

    void set_interval (uint32_t inteval) {
        _interval = inteval;
    }

    void set_frame_width (uint32_t frame_width) {
        _frame_width = frame_width;
    }

    void set_frame_height (uint32_t frame_height) {
        _frame_height = frame_height;
    }

    void set_frame_save (uint32_t frame_save) {
        _frame_save = frame_save;
    }

    void enable_display(bool value) {
        _enable_display = value;
    }

    void set_display_mode(DrmDisplayMode mode) {
        _display->set_display_mode (mode);
    }

protected:
    virtual void handle_message (const SmartPtr<XCamMessage> &msg);
    virtual void handle_buffer (const SmartPtr<VideoBuffer> &buf);

    int display_buf (const SmartPtr<VideoBuffer> &buf);

private:
    void open_file ();
    void close_file ();
    XCamReturn write_buf (const SmartPtr<VideoBuffer> &buf);

    FILE      *_file;
    bool       _save_file;
    uint32_t   _interval;
    uint32_t   _frame_width;
    uint32_t   _frame_height;
    uint32_t   _frame_count;
    uint32_t   _frame_save;
    SmartPtr<DrmDisplay> _display;
    bool       _enable_display;
    XCAM_OBJ_PROFILING_DEFINES;
};

void
MainDeviceManager::handle_message (const SmartPtr<XCamMessage> &msg)
{
    XCAM_UNUSED (msg);
}

void
MainDeviceManager::handle_buffer (const SmartPtr<VideoBuffer> &buf)
{
    FPS_CALCULATION (fps_buf, 30);

    XCAM_OBJ_PROFILING_START;

    if (_enable_display)
        display_buf (buf);

    XCAM_OBJ_PROFILING_END("main_dev_manager_display", 30);

    if (!_save_file)
        return ;

    if ((_frame_count++ % _interval) != 0)
        return;

    if ((_frame_save != 0) && (_frame_count > _frame_save)) {
        SmartLock locker (g_mutex);
        g_stop = true;
        g_cond.broadcast ();
        return;
    }

    open_file ();

    if (!_file) {
        XCAM_LOG_ERROR ("open file failed");
        return;
    }
    write_buf (buf);
}

int
MainDeviceManager::display_buf (const SmartPtr<VideoBuffer> &data)
{
#if HAVE_LIBDRM
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<VideoBuffer> buf = data;
    const VideoBufferInfo & frame_info = buf->get_video_info ();
    struct v4l2_rect rect = { 0, 0, (int)frame_info.width, (int)frame_info.height };

    if (!_display->is_render_inited ()) {
        ret = _display->render_init (0, 0, this->_frame_width, this->_frame_height,
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


void
MainDeviceManager::open_file ()
{
    if ((_file) && (_frame_save == 0))
        return;

    std::string file_name = DEFAULT_SAVE_FILE_NAME;

    if (_frame_save != 0) {
        file_name += std::to_string(_frame_count);
    }
    file_name += ".raw";

    _file = fopen(file_name.c_str(), "wb");
}

void
MainDeviceManager::close_file ()
{
    if (_file)
        fclose (_file);
    _file = NULL;
}

XCamReturn
MainDeviceManager::write_buf (const SmartPtr<VideoBuffer> &buf)
{
    const VideoBufferInfo &info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    memory = buf->map ();
    if (!memory) {
        XCAM_LOG_ERROR ("map buffer failed in write_buf");
        return XCAM_RETURN_ERROR_MEM;
    }

    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fwrite (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, _file) != line_bytes) {
                XCAM_LOG_ERROR ("write file failed, size doesn't match");
                ret = XCAM_RETURN_ERROR_FILE;
            }
        }
    }
    buf->unmap ();
    return ret;
}

#define V4L2_CAPTURE_MODE_STILL   0x2000
#define V4L2_CAPTURE_MODE_VIDEO   0x4000
#define V4L2_CAPTURE_MODE_PREVIEW 0x8000

typedef enum {
    AnalyzerTypeSimple = 0,
    AnalyzerTypeAiqTuner,
    AnalyzerTypeDynamic,
    AnalyzerTypeHybrid,
} AnalyzerType;

void dev_stop_handler(int sig)
{
    XCAM_UNUSED (sig);

    SmartLock locker (g_mutex);
    g_stop = true;
    g_cond.broadcast ();

    //exit(0);
}

void print_help (const char *bin_name)
{
    printf ("Usage: %s [-a analyzer]\n"
            "Configurations:\n"
            "\t -a analyzer   specify a analyzer\n"
            "\t               select from [simple, aiq, dynamic], default is [simple]\n"
            "\t -m mem_type   specify video memory type\n"
            "\t               mem_type select from [dma, mmap], default is [mmap]\n"
            "\t -s            save file to %s\n"
            "\t -n interval   save file on every [interval] frame\n"
#if HAVE_LIBCL
            "\t -c            process image with cl kernel\n"
#endif
            "\t -f pixel_fmt  specify output pixel format\n"
            "\t               pixel_fmt select from [NV12, YUYV, BA10, BA12], default is [NV12]\n"
            "\t -d cap_mode   specify capture mode\n"
            "\t               cap_mode select from [video, still], default is [video]\n"
            "\t -b brightness specify brightness level\n"
            "\t               brightness level select from [0, 256], default is [128]\n"
            "\t -i frame_save specify the frame count to save, default is 0 which means endless\n"
            "\t -p preview on local display\n"
            "\t --usb         specify node for usb camera device, enables capture path through USB camera \n"
            "\t               specify [/dev/video4, /dev/video5] depending on which node USB camera is attached\n"
            "\t --resolution  specify the resolution of usb camera\n"
            "\t               select from [1920x1080, 1280x720 ...], default is [1920x1080]\n"
            "\t -e display_mode    preview mode\n"
            "\t                select from [primary, overlay], default is [primary]\n"
            "\t --sync        set analyzer in sync mode\n"
            "\t -r raw_input  specify the path of raw image as fake source instead of live camera\n"
            "\t -h            help\n"
#if HAVE_LIBCL
            "CL features:\n"
            "\t --capture capture_stage      specify the capture stage of image\n"
            "\t               capture_stage select from [bayer, tonemapping], default is [tonemapping]\n"
            "\t --hdr         specify hdr type, default is hdr off\n"
            "\t               select from [rgb, lab]\n"
            "\t --tnr         specify temporal noise reduction type, default is tnr off\n"
            "\t               select from [rgb, yuv, both]\n"
            "\t --tnr-level   specify tnr level\n"
            "\t --wdr-mode    specify wdr mode. select from [gaussian, haleq]\n"
            "\t --bilateral   enable bilateral noise reduction\n"
            "\t --enable-snr  enable simple noise reduction\n"
            "\t --enable-ee   enable YEENR\n"
            "\t --enable-bnr  enable bayer noise reduction\n"
            "\t --enable-dpc  enable defect pixel correction\n"
            "\t --defog-mode mode   enable defog\n"
            "\t               select from [disabled, retinex, dcp], default is [disabled]\n"
            "\t --enable-retinex  enable retinex\n"
            "\t --wavelet-mode specify wavelet denoise mode, default is off\n"
            "\t                select from [0:disable, 1:Hat Y, 2:Hat UV, 3:Haar Y, 4:Haar UV, 5:Haar YUV, 6:Haar Bayes Shrink]\n"
            "\t --enable-wireframe  enable wire frame\n"
            "\t --pipeline    pipe mode\n"
            "\t               select from [basic, advance, extreme], default is [basic]\n"
            "\t --disable-post disable cl post image processor\n"
            "(e.g.: xxxx --hdr=xx --tnr=xx --tnr-level=xx --bilateral --enable-snr --enable-ee --enable-bnr --enable-dpc)\n\n"
#endif
            , bin_name
            , DEFAULT_SAVE_FILE_NAME);
}

int main (int argc, char *argv[])
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<MainDeviceManager> device_manager = new MainDeviceManager;
    SmartPtr<V4l2Device> device;
    SmartPtr<V4l2SubDevice> event_device;
    SmartPtr<IspController> isp_controller;
    SmartPtr<X3aAnalyzer> analyzer;
    SmartPtr<SmartAnalyzer> smart_analyzer;
    SmartPtr<AnalyzerLoader> loader;
    const char *path_of_3a;
    SmartPtr<ImageProcessor> isp_processor;
    AnalyzerType  analyzer_type = AnalyzerTypeSimple;
    DrmDisplayMode display_mode = DRM_DISPLAY_MODE_PRIMARY;
#if HAVE_LIBDRM
    SmartPtr<DrmDisplay> drm_disp = DrmDisplay::instance();
#endif

#if HAVE_LIBCL
    SmartPtr<CL3aImageProcessor> cl_processor;
    SmartPtr<CLPostImageProcessor> cl_post_processor;
    uint32_t hdr_type = CL_HDR_DISABLE;
    uint32_t tnr_type = CL_TNR_DISABLE;
    uint32_t denoise_type = 0;
    uint8_t tnr_level = 0;
    bool dpc_type = false;
    CL3aImageProcessor::PipelineProfile pipeline_mode = CL3aImageProcessor::BasicPipelineProfile;
    CL3aImageProcessor::CaptureStage capture_stage = CL3aImageProcessor::TonemappingStage;
    CL3aImageProcessor::CLTonemappingMode wdr_mode = CL3aImageProcessor::WDRdisabled;
#endif
    bool have_cl_processor = false;
    bool have_cl_post_processor = true;
    bool need_display = false;
    enum v4l2_memory v4l2_mem_type = V4L2_MEMORY_MMAP;
    const char *bin_name = argv[0];
    int opt;
    uint32_t capture_mode = V4L2_CAPTURE_MODE_VIDEO;
    uint32_t pixel_format = V4L2_PIX_FMT_NV12;
    bool wdr_type = false;
    uint32_t defog_type = 0;
    CLWaveletBasis wavelet_mode = CL_WAVELET_DISABLED;
    uint32_t wavelet_channel = CL_IMAGE_CHANNEL_UV;
    bool wavelet_bayes_shrink = false;
    bool wireframe_type = false;

    int32_t brightness_level = 128;
    bool    have_usbcam = 0;
    SmartPtr<char> usb_device_name;
    bool sync_mode = false;
    int frame_rate;
    int frame_width = 1920;
    int frame_height = 1080;
    SmartPtr<char> path_to_fake = NULL;

    const char *short_opts = "sca:n:m:f:d:b:pi:e:r:h";
    const struct option long_opts[] = {
        {"hdr", required_argument, NULL, 'H'},
        {"tnr", required_argument, NULL, 'T'},
        {"tnr-level", required_argument, NULL, 'L'},
        {"wdr-mode", required_argument, NULL, 'W'},
        {"bilateral", no_argument, NULL, 'I'},
        {"enable-snr", no_argument, NULL, 'S'},
        {"enable-ee", no_argument, NULL, 'E'},
        {"enable-bnr", no_argument, NULL, 'B'},
        {"enable-dpc", no_argument, NULL, 'D'},
        {"defog-mode", required_argument, NULL, 'X'},
        {"wavelet-mode", required_argument, NULL, 'V'},
        {"enable-wireframe", no_argument, NULL, 'F'},
        {"usb", required_argument, NULL, 'U'},
        {"resolution", required_argument, NULL, 'R'},
        {"sync", no_argument, NULL, 'Y'},
        {"capture", required_argument, NULL, 'C'},
        {"pipeline", required_argument, NULL, 'P'},
        {"disable-post", no_argument, NULL, 'O'},
        {0, 0, 0, 0},
    };

    while ((opt = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'a': {
            XCAM_ASSERT (optarg);
            if (!strcmp (optarg, "dynamic"))
                analyzer_type = AnalyzerTypeDynamic;
            else if (!strcmp (optarg, "simple"))
                analyzer_type = AnalyzerTypeSimple;
#if HAVE_IA_AIQ
            else if (!strcmp (optarg, "aiq"))
                analyzer_type = AnalyzerTypeAiqTuner;
            else if (!strcmp (optarg, "hybrid"))
                analyzer_type = AnalyzerTypeHybrid;
#endif
            else {
                print_help (bin_name);
                return -1;
            }
            break;
        }

        case 'm': {
            XCAM_ASSERT (optarg);
            if (!strcmp (optarg, "dma"))
                v4l2_mem_type = V4L2_MEMORY_DMABUF;
            else if (!strcmp (optarg, "mmap"))
                v4l2_mem_type = V4L2_MEMORY_MMAP;
            else
                print_help (bin_name);
            break;
        }

        case 's':
            device_manager->enable_save_file (true);
            break;
        case 'n':
            XCAM_ASSERT (optarg);
            device_manager->set_interval (atoi(optarg));
            break;
#if HAVE_LIBCL
        case 'c':
            have_cl_processor = true;
            break;
#endif
        case 'f':
            XCAM_ASSERT (optarg);
            CHECK_EXP ((strlen(optarg) == 4), "invalid pixel format\n");
            pixel_format = v4l2_fourcc ((unsigned)optarg[0],
                                        (unsigned)optarg[1],
                                        (unsigned)optarg[2],
                                        (unsigned)optarg[3]);
            break;
        case 'd':
            XCAM_ASSERT (optarg);
            if (!strcmp (optarg, "still"))
                capture_mode = V4L2_CAPTURE_MODE_STILL;
            else if (!strcmp (optarg, "video"))
                capture_mode = V4L2_CAPTURE_MODE_VIDEO;
            else  {
                print_help (bin_name);
                return -1;
            }
            break;
        case 'b':
            XCAM_ASSERT (optarg);
            brightness_level = atoi(optarg);
            if(brightness_level < 0 || brightness_level > 256) {
                print_help (bin_name);
                return -1;
            }
            break;
        case 'p':
            need_display = true;
            break;
        case 'U':
            XCAM_ASSERT (optarg);
            have_usbcam = true;
            usb_device_name = strndup(optarg, XCAM_MAX_STR_SIZE);
            XCAM_LOG_DEBUG("using USB camera plugged in at node: %s", XCAM_STR(usb_device_name.ptr ()));
            break;
        case 'R':
            XCAM_ASSERT (optarg);
            sscanf (optarg, "%d%*c%d", &frame_width, &frame_height);
            break;
        case 'e': {
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
        case 'i':
            XCAM_ASSERT (optarg);
            device_manager->set_frame_save(atoi(optarg));
            break;
        case 'Y':
            sync_mode = true;
            break;
#if HAVE_LIBCL
        case 'H': {
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "rgb"))
                hdr_type = CL_HDR_TYPE_RGB;
            else if (!strcasecmp (optarg, "lab"))
                hdr_type = CL_HDR_TYPE_LAB;
            else {
                print_help (bin_name);
                return -1;
            }
            break;
        }
        case 'I': {
            denoise_type |= XCAM_DENOISE_TYPE_BILATERAL;
            denoise_type |= XCAM_DENOISE_TYPE_BIYUV;
            break;
        }
        case 'S': {
            denoise_type |= XCAM_DENOISE_TYPE_SIMPLE;
            break;
        }
        case 'E': {
            denoise_type |= XCAM_DENOISE_TYPE_EE;
            break;
        }
        case 'B': {
            denoise_type |= XCAM_DENOISE_TYPE_BNR;
            break;
        }
        case 'X': {
            XCAM_ASSERT (optarg);
            defog_type = true;
            if (!strcmp (optarg, "disabled"))
                defog_type = CLPostImageProcessor::DefogDisabled;
            else if (!strcmp (optarg, "retinex"))
                defog_type = CLPostImageProcessor::DefogRetinex;
            else if (!strcmp (optarg, "dcp"))
                defog_type = CLPostImageProcessor::DefogDarkChannelPrior;
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
        case 'F': {
            wireframe_type = true;
            break;
        }
        case 'D': {
            dpc_type = true;
            break;
        }
        case 'T': {
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "yuv"))
                tnr_type = CL_TNR_TYPE_YUV;
            else if (!strcasecmp (optarg, "rgb"))
                tnr_type = CL_TNR_TYPE_RGB;
            else if (!strcasecmp (optarg, "both"))
                tnr_type = CL_TNR_TYPE_YUV | CL_TNR_TYPE_RGB;
            else {
                print_help (bin_name);
                return -1;
            }
            break;
        }
        case 'L': {
            XCAM_ASSERT (optarg);
            if (atoi(optarg) < 0 || atoi(optarg) > 255) {
                print_help (bin_name);
                return -1;
            }
            tnr_level = atoi(optarg);
            break;
        }
        case 'W': {
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "gaussian"))
                wdr_mode = CL3aImageProcessor::Gaussian;
            else if (!strcasecmp (optarg, "haleq"))
                wdr_mode = CL3aImageProcessor::Haleq;

            pixel_format = V4L2_PIX_FMT_SGRBG12;
            wdr_type = true;
            setenv ("AIQ_CPF_PATH", IMX185_WDR_CPF, 1);
            break;
        }
        case 'P': {
            XCAM_ASSERT (optarg);
            if (!strcasecmp (optarg, "basic"))
                pipeline_mode = CL3aImageProcessor::BasicPipelineProfile;
            else if (!strcasecmp (optarg, "advance"))
                pipeline_mode = CL3aImageProcessor::AdvancedPipelineProfile;
            else if (!strcasecmp (optarg, "extreme"))
                pipeline_mode = CL3aImageProcessor::ExtremePipelineProfile;
            else {
                print_help (bin_name);
                return -1;
            }
            break;
        }
        case 'C': {
            XCAM_ASSERT (optarg);
            if (!strcmp (optarg, "bayer"))
                capture_stage = CL3aImageProcessor::BasicbayerStage;
            break;
        }
        case 'O': {
            have_cl_post_processor = false;
            break;
        }
#endif
        case 'r': {
            XCAM_ASSERT (optarg);
            XCAM_LOG_INFO ("use raw image %s as input source", optarg);
            path_to_fake = strndup(optarg, XCAM_MAX_STR_SIZE);
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

    device_manager->set_frame_width(frame_width);
    device_manager->set_frame_height(frame_height);
    if (need_display) {
        device_manager->enable_display (true);
        device_manager->set_display_mode (display_mode);
    }
    if (!device.ptr ())  {
        if (path_to_fake.ptr ()) {
            device = new FakeV4l2Device ();
        } else if (have_usbcam) {
            device = new UVCDevice (usb_device_name.ptr ());
        } else {
            if (capture_mode == V4L2_CAPTURE_MODE_STILL)
                device = new AtomispDevice (CAPTURE_DEVICE_STILL);
            else if (capture_mode == V4L2_CAPTURE_MODE_VIDEO)
                device = new AtomispDevice (CAPTURE_DEVICE_VIDEO);
            else
                device = new AtomispDevice (DEFAULT_CAPTURE_DEVICE);
        }
    }
    if (!event_device.ptr ())
        event_device = new V4l2SubDevice (DEFAULT_EVENT_DEVICE);
    if (!isp_controller.ptr ())
        isp_controller = new IspController (device);

    switch (analyzer_type) {
    case AnalyzerTypeSimple:
        analyzer = new X3aAnalyzerSimple ();
        break;
#if HAVE_IA_AIQ
    case AnalyzerTypeAiqTuner: {
        SmartPtr<X3aAnalyzer> aiq_analyzer = new X3aAnalyzerAiq (isp_controller, DEFAULT_CPF_FILE);
        SmartPtr<X3aAnalyzeTuner> tuner_analyzer = new X3aAnalyzeTuner ();
        XCAM_ASSERT (aiq_analyzer.ptr () && tuner_analyzer.ptr ());
        tuner_analyzer->set_analyzer (aiq_analyzer);
        analyzer = tuner_analyzer;
        break;
    }
    case AnalyzerTypeHybrid: {
        path_of_3a = DEFAULT_HYBRID_3A_LIB;
        SmartPtr<HybridAnalyzerLoader> hybrid_loader = new HybridAnalyzerLoader (path_of_3a);
        hybrid_loader->set_cpf_path (DEFAULT_CPF_FILE);
        hybrid_loader->set_isp_controller (isp_controller);
        loader = hybrid_loader.dynamic_cast_ptr<AnalyzerLoader> ();
        analyzer = hybrid_loader->load_analyzer (loader);
        CHECK_EXP (analyzer.ptr (), "load hybrid 3a lib(%s) failed", path_of_3a);
        break;
    }
#endif
    case AnalyzerTypeDynamic: {
        path_of_3a = DEFAULT_DYNAMIC_3A_LIB;
        SmartPtr<DynamicAnalyzerLoader> dynamic_loader = new DynamicAnalyzerLoader (path_of_3a);
        loader = dynamic_loader.dynamic_cast_ptr<AnalyzerLoader> ();
        analyzer = dynamic_loader->load_analyzer (loader);
        CHECK_EXP (analyzer.ptr (), "load dynamic 3a lib(%s) failed", path_of_3a);

        // Create smart analyzer from dynamic libraries
        SmartHandlerList smart_handlers = SmartAnalyzerLoader::load_smart_handlers (DEFAULT_SMART_ANALYSIS_LIB_DIR);
        if (!smart_handlers.empty () ) {
            smart_analyzer = new SmartAnalyzer ();
            if (!smart_analyzer.ptr ()) {
                XCAM_LOG_INFO ("load smart analyzer(%s) failed", DEFAULT_SMART_ANALYSIS_LIB_DIR);
                break;
            }
            SmartHandlerList::iterator i_handler = smart_handlers.begin ();
            for (; i_handler != smart_handlers.end (); ++i_handler) {
                XCAM_ASSERT ((*i_handler).ptr ());
                smart_analyzer->add_handler (*i_handler);
            }
        }
        break;
    }
    default:
        print_help (bin_name);
        return -1;
    }
    XCAM_ASSERT (analyzer.ptr ());
    analyzer->set_sync_mode (sync_mode);

    signal(SIGINT, dev_stop_handler);

    device->set_sensor_id (0);
    device->set_capture_mode (capture_mode);
    //device->set_mem_type (V4L2_MEMORY_DMABUF);
    device->set_mem_type (v4l2_mem_type);
    device->set_buffer_count (8);
    if (pixel_format == V4L2_PIX_FMT_SGRBG12) {
        frame_rate = 30;
        device->set_framerate (frame_rate, 1);
    }
    else {
        frame_rate = 25;
        device->set_framerate (frame_rate, 1);
        if(wdr_type == true) {
            XCAM_LOG_WARNING("Tonemapping is only applicable under BA12 format. Disable tonemapping automatically.");
            wdr_type = false;
        }
    }
    ret = device->open ();
    CHECK (ret, "device(%s) open failed", device->get_device_name());
    ret = device->set_format (frame_width, frame_height, pixel_format, V4L2_FIELD_NONE, frame_width * 2);
    CHECK (ret, "device(%s) set format failed", device->get_device_name());

    ret = event_device->open ();
    if (ret == XCAM_RETURN_NO_ERROR) {
        CHECK (ret, "event device(%s) open failed", event_device->get_device_name());
        int event = V4L2_EVENT_ATOMISP_3A_STATS_READY;
        ret = event_device->subscribe_event (event);
        CHECK_CONTINUE (
            ret,
            "device(%s) subscribe event(%d) failed",
            event_device->get_device_name(), event);
        event = V4L2_EVENT_FRAME_SYNC;
        ret = event_device->subscribe_event (event);
        CHECK_CONTINUE (
            ret,
            "device(%s) subscribe event(%d) failed",
            event_device->get_device_name(), event);

        device_manager->set_event_device (event_device);
    }

    device_manager->set_capture_device (device);
    if (analyzer.ptr())
        device_manager->set_3a_analyzer (analyzer);

    if (smart_analyzer.ptr ()) {
        if (smart_analyzer->prepare_handlers () != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("analyzer(%s) prepare handlers failed", smart_analyzer->get_name ());
        }
        device_manager->set_smart_analyzer (smart_analyzer);
    }

    if (have_cl_processor)
        isp_processor = new IspExposureImageProcessor (isp_controller);
    else
        isp_processor = new IspImageProcessor (isp_controller);

    XCAM_ASSERT (isp_processor.ptr ());
    device_manager->add_image_processor (isp_processor);
#if HAVE_LIBCL
    if (have_cl_processor) {
        cl_processor = new CL3aImageProcessor ();
        cl_processor->set_stats_callback(device_manager);
        cl_processor->set_dpc(dpc_type);
        cl_processor->set_hdr (hdr_type);
        cl_processor->set_denoise (denoise_type);
        cl_processor->set_tonemapping(wdr_mode);
        cl_processor->set_gamma (!wdr_type); // disable gamma for WDR
        cl_processor->set_wavelet (wavelet_mode, wavelet_channel, wavelet_bayes_shrink);
        cl_processor->set_wireframe (wireframe_type);
        cl_processor->set_capture_stage (capture_stage);

        if (wdr_type) {
            cl_processor->set_3a_stats_bits(12);
        }
        cl_processor->set_tnr (tnr_type, tnr_level);
        cl_processor->set_profile (pipeline_mode);
        analyzer->set_parameter_brightness((brightness_level - 128) / 128.0);
        device_manager->add_image_processor (cl_processor);

        if (smart_analyzer.ptr ()) {
            cl_processor->set_scaler (true);
            cl_processor->set_scaler_factor (640.0 / frame_width);
        }
    }

    if (have_cl_post_processor) {
        cl_post_processor = new CLPostImageProcessor ();

        cl_post_processor->set_defog_mode ((CLPostImageProcessor::CLDefogMode)defog_type);

        if (need_display) {
            cl_post_processor->set_output_format (V4L2_PIX_FMT_XBGR32);
        }
        device_manager->add_image_processor (cl_post_processor);
    }
#endif

    SmartPtr<PollThread> poll_thread;
    if (path_to_fake.ptr ())
        poll_thread = new FakePollThread (path_to_fake.ptr ());
    else {
        SmartPtr<IspPollThread> isp_poll_thread = new IspPollThread ();
        isp_poll_thread->set_isp_controller (isp_controller);
        poll_thread = isp_poll_thread;
    }
    device_manager->set_poll_thread (poll_thread);

    ret = device_manager->start ();
    CHECK (ret, "device manager start failed");

    // hard code exposure range and max gain for imx185 WDR
    if (wdr_type) {
        if (frame_rate == 30)
            analyzer->set_ae_exposure_time_range (80 * 1110 * 1000 / 37125, 1120 * 1110 * 1000 / 37125);
        else
            analyzer->set_ae_exposure_time_range (80 * 1320 * 1000 / 37125, 1120 * 1320 * 1000 / 37125);
        analyzer->set_ae_max_analog_gain (3.98); // 12dB
    }

    // wait for interruption
    {
        SmartLock locker (g_mutex);
        while (!g_stop)
            g_cond.wait (g_mutex);
    }

    ret = device_manager->stop();
    CHECK_CONTINUE (ret, "device manager stop failed");
    device->close ();
    event_device->close ();

    return 0;
}
