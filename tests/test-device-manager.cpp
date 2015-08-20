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
#include "isp_controller.h"
#include "isp_image_processor.h"
#include "x3a_analyzer_simple.h"
#if HAVE_IA_AIQ
#include "x3a_analyzer_aiq.h"
#endif
#if HAVE_LIBCL
#include "cl_3a_image_processor.h"
#include "cl_csc_image_processor.h"
#include "cl_hdr_handler.h"
#include "cl_tnr_handler.h"
#endif
#if HAVE_LIBDRM
#include "drm_display.h"
#endif
#include "analyzer_loader.h"
#include <base/xcam_3a_types.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>
#include <string>
#include <getopt.h>
#include "test_common.h"

using namespace XCam;

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

    FILE      *_file;
    bool       _save_file;
    uint32_t   _interval;
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

    const VideoBufferInfo & frame_info = buf->get_video_info ();
    uint8_t *frame = buf->map ();

    if (frame == NULL)
        return;

    uint32_t size = 0;

    switch(frame_info.format)  {
    case V4L2_PIX_FMT_NV12:  // 420
    case V4L2_PIX_FMT_NV21:
        size = XCAM_ALIGN_UP(frame_info.width, 2) * XCAM_ALIGN_UP(frame_info.height, 2) * 3 / 2;
        break;
    case V4L2_PIX_FMT_YUV422P: // 422 Planar
    case V4L2_PIX_FMT_YUYV: // 422
    case V4L2_PIX_FMT_SBGGR10:
    case V4L2_PIX_FMT_SGBRG10:
    case V4L2_PIX_FMT_SGRBG10:
    case V4L2_PIX_FMT_SRGGB10:
    case V4L2_PIX_FMT_SBGGR12:
    case V4L2_PIX_FMT_SGBRG12:
    case V4L2_PIX_FMT_SGRBG12:
    case V4L2_PIX_FMT_SRGGB12:
        size = XCAM_ALIGN_UP(frame_info.width, 2) * XCAM_ALIGN_UP(frame_info.height, 2) * 2;
        break;
    default:
        XCAM_LOG_ERROR (
            "unknown v4l2 format(%s) in buffer handle",
            xcam_fourcc_to_string (frame_info.format));
        return;
    }

    open_file ();

    if (!_file) {
        XCAM_LOG_ERROR ("open file failed");
        return;
    }

    if (fwrite (frame, size, 1, _file) <= 0) {
        XCAM_LOG_WARNING ("write frame failed.");
    }
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
        ret = _display->render_init (0, 0, 1920, 1080, frame_info.format, &rect);
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

#define V4L2_CAPTURE_MODE_STILL   0x2000
#define V4L2_CAPTURE_MODE_VIDEO   0x4000
#define V4L2_CAPTURE_MODE_PREVIEW 0x8000

typedef enum {
    AnalyzerTypeSimple = 0,
    AnalyzerTypeAiq,
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
            "\t -c            process image with cl kernel\n"
            "\t -f pixel_fmt  specify output pixel format\n"
            "\t               pixel_fmt select from [NV12, YUYV, BA10, RG12], default is [NV12]\n"
            "\t -d cap_mode   specify capture mode\n"
            "\t               cap_mode select from [video, still], default is [video]\n"
            "\t -b brightness specify brightness level\n"
            "\t               brightness level select from [0, 256], default is [128]\n"
            "\t -i frame_save specify the frame count to save, default is 0 which means endless\n"
            "\t -p preview on local display\n"
            "\t --usb         specify node for usb camera device, enables capture path through USB camera \n"
            "\t               specify [/dev/video4, /dev/video5] depending on which node USB camera is attached\n"
            "\t -e display_mode    preview mode\n"
            "\t                select from [primary, overlay], default is [primary]\n"
            "\t --sync        set analyzer in sync mode\n"
            "\t -h            help\n"
            "CL features:\n"
            "\t --hdr         specify hdr type, default is hdr off\n"
            "\t               select from [rgb, lab]\n"
            "\t --tnr         specify temporal noise reduction type, default is tnr off\n"
            "\t               select from [rgb, yuv, both]\n"
            "\t --tnr-level   specify tnr level\n"
            "\t --bilateral   enable bilateral noise reduction\n"
            "\t --enable-snr  enable simple noise reduction\n"
            "\t --enable-ee   enable YEENR\n"
            "\t --enable-bnr  enable bayer noise reduction\n"
            "\t --enable-dpc  enable defect pixel correction\n"
            "\t --enable-tonemapping  enable tonemapping\n"
            "(e.g.: xxxx --hdr=xx --tnr=xx --tnr-level=xx --bilateral --enable-snr --enable-ee --enable-bnr --enable-dpc)\n\n"
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
    SmartPtr<AnalyzerLoader> loader;
    const char *path_of_3a;
    SmartPtr<ImageProcessor> isp_processor;
    SmartPtr<CLCscImageProcessor> cl_csc_proccessor;
    AnalyzerType  analyzer_type = AnalyzerTypeSimple;
    DrmDisplayMode display_mode = DRM_DISPLAY_MODE_PRIMARY;
#if HAVE_LIBDRM
    SmartPtr<DrmDisplay> drm_disp = DrmDisplay::instance();
#endif

#if HAVE_LIBCL
    SmartPtr<CL3aImageProcessor> cl_processor;
#endif
    bool have_cl_processor = false;
    bool need_display = false;
    enum v4l2_memory v4l2_mem_type = V4L2_MEMORY_MMAP;
    const char *bin_name = argv[0];
    int opt;
    uint32_t capture_mode = V4L2_CAPTURE_MODE_VIDEO;
    uint32_t pixel_format = V4L2_PIX_FMT_NV12;
    uint32_t hdr_type = CL_HDR_DISABLE;
    uint32_t tnr_type = CL_TNR_DISABLE;
    uint32_t denoise_type = 0;
    uint8_t tnr_level = 0;
    bool dpc_type = false;
    bool tonemapping_type = false;
    int32_t brightness_level = 128;
    bool    have_usbcam = 0;
    char*   usb_device_name = NULL;
    bool sync_mode = false;

    const char *short_opts = "sca:n:m:f:d:b:pi:e:h";
    const struct option long_opts[] = {
        {"hdr", required_argument, NULL, 'H'},
        {"tnr", required_argument, NULL, 'T'},
        {"tnr-level", required_argument, NULL, 'L'},
        {"bilateral", no_argument, NULL, 'I'},
        {"enable-snr", no_argument, NULL, 'S'},
        {"enable-ee", no_argument, NULL, 'E'},
        {"enable-bnr", no_argument, NULL, 'B'},
        {"enable-dpc", no_argument, NULL, 'D'},
        {"enable-tonemapping", no_argument, NULL, 'M'},
        {"usb", required_argument, NULL, 'U'},
        {"sync", no_argument, NULL, 'Y'},
        {0, 0, 0, 0},
    };

    while ((opt = getopt_long(argc, argv, short_opts, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'a': {
            if (!strcmp (optarg, "dynamic"))
                analyzer_type = AnalyzerTypeDynamic;
            else if (!strcmp (optarg, "simple"))
                analyzer_type = AnalyzerTypeSimple;
#if HAVE_IA_AIQ
            else if (!strcmp (optarg, "aiq"))
                analyzer_type = AnalyzerTypeAiq;
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
            device_manager->set_interval (atoi(optarg));
            break;
        case 'c':
            have_cl_processor = true;
            break;
        case 'f':
            CHECK_EXP ((strlen(optarg) == 4), "invalid pixel format\n");
            pixel_format = v4l2_fourcc ((unsigned)optarg[0],
                                        (unsigned)optarg[1],
                                        (unsigned)optarg[2],
                                        (unsigned)optarg[3]);
            break;
        case 'd':
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
            have_usbcam = true;
            usb_device_name = strdup(optarg);
            XCAM_LOG_DEBUG("using USB camera plugged in at node: %s", usb_device_name);
            break;
        case 'e': {
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
            device_manager->set_frame_save(atoi(optarg));
            break;
        case 'Y':
            sync_mode = true;
            break;
        case 'H': {
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
        case 'D': {
            dpc_type = true;
            break;
        }
        case 'T': {
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
            if (atoi(optarg) < 0 || atoi(optarg) > 255) {
                print_help (bin_name);
                return -1;
            }
            tnr_level = atoi(optarg);
            break;
        }
        case 'M': {
            tonemapping_type = true;
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

    if (need_display) {
        device_manager->enable_display (true);
        device_manager->set_display_mode (display_mode);
    }
    if (!device.ptr ())  {
        if (have_usbcam) {
            device = new UVCDevice (usb_device_name);
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
    case AnalyzerTypeAiq:
        analyzer = new X3aAnalyzerAiq (isp_controller, DEFAULT_CPF_FILE);
        break;
    case AnalyzerTypeHybrid: {
        path_of_3a = DEFAULT_DYNAMIC_3A_LIB;
        loader = new AnalyzerLoader (path_of_3a);
        analyzer = loader->load_hybrid_analyzer (loader, isp_controller, DEFAULT_CPF_FILE);
        CHECK_EXP (analyzer.ptr (), "load hybrid 3a lib(%s) failed", path_of_3a);
        break;
    }
#endif
    case AnalyzerTypeDynamic: {
        path_of_3a = DEFAULT_DYNAMIC_3A_LIB;
        loader = new AnalyzerLoader (path_of_3a);
        analyzer = loader->load_dynamic_analyzer (loader);
        CHECK_EXP (analyzer.ptr (), "load dynamic 3a lib(%s) failed", path_of_3a);
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
    if (pixel_format == V4L2_PIX_FMT_SRGGB12)
        device->set_framerate (30, 1);
    else
        device->set_framerate (25, 1);
    ret = device->open ();
    CHECK (ret, "device(%s) open failed", device->get_device_name());
    ret = device->set_format (1920, 1080, pixel_format, V4L2_FIELD_NONE, 1920 * 2);
    CHECK (ret, "device(%s) set format failed", device->get_device_name());

    ret = event_device->open ();
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

    device_manager->set_capture_device (device);
    device_manager->set_event_device (event_device);
    device_manager->set_isp_controller (isp_controller);
    if (analyzer.ptr())
        device_manager->set_analyzer (analyzer);

    if (have_cl_processor)
        isp_processor = new IspExposureImageProcessor (isp_controller);
    else
        isp_processor = new IspImageProcessor (isp_controller);

    XCAM_ASSERT (isp_processor.ptr ());
    device_manager->add_image_processor (isp_processor);
    if ((display_mode == DRM_DISPLAY_MODE_PRIMARY) && need_display && (!have_cl_processor)) {
        cl_csc_proccessor = new CLCscImageProcessor();
        XCAM_ASSERT (cl_csc_proccessor.ptr ());
        device_manager->add_image_processor (cl_csc_proccessor);
    }

#if HAVE_LIBCL
    if (have_cl_processor) {

        cl_processor = new CL3aImageProcessor ();
        cl_processor->set_stats_callback(device_manager);
        cl_processor->set_dpc(dpc_type);
        cl_processor->set_hdr (hdr_type);
        cl_processor->set_denoise (denoise_type);
        cl_processor->set_tonemapping(tonemapping_type);
        if (need_display) {
            cl_processor->set_output_format (V4L2_PIX_FMT_XBGR32);
        }
        cl_processor->set_tnr (tnr_type, tnr_level);
        analyzer->set_parameter_brightness((brightness_level - 128) / 128.0);
        device_manager->add_image_processor (cl_processor);
    }
#endif

    ret = device_manager->start ();
    CHECK (ret, "device manager start failed");

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
