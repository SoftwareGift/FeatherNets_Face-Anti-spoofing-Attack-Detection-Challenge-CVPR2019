/*
 * test-image-stitching.cpp - test image stitching
 *
 *  Copyright (c) 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * 	 http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "test_common.h"
#include <unistd.h>
#include <getopt.h>
#include "cl_device.h"
#include "cl_context.h"
#include "drm_display.h"
#include "image_file_handle.h"

#include "cl_fisheye_handler.h"
#include "cl_blender.h"
#include "cl_image_360_stitch.h"

using namespace XCam;

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t top;
    uint32_t bottom;
} FisheyeCropInfo;

static FisheyeCropInfo fisheye_crop0 = {
    .left = 92,
    .right = 108,
    .top = 0,
    .bottom = 0,
};
static FisheyeCropInfo fisheye_crop1 = {
    .left = 98,
    .right = 106,
    .top = 0,
    .bottom = 0,
};

static const float max_dst_angle = 230.0f;
static uint32_t merge_width0 = 62;
static uint32_t merge_width1 = 58;

enum ImageStitchMode {
    IMAGE_STITCH_MODE_360 = 0,
    IMAGE_STITCH_MODE_BLEND
};

struct ImageStitchInfo {
    Rect stitch_left;
    Rect stitch_right;
};

void usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input file --output file\n"
            "\t--input       input image(NV12)\n"
            "\t--output      output image(NV12)\n"
            "\t--input-w     optional, input width, default: 1920\n"
            "\t--input-h     optional, input height, default: 1080\n"
            "\t--output-w    optional, output width, default: 1920\n"
            "\t--output-h    optional, output width, default: 960\n"
            "\t--loop        optional, how many loops need to run for performance test, default: 0\n"
            "\t--save        optional, save file or not, select from [true/false], default: true\n"
            "\t--stitch-mode optional, image stitching mode, select from [360/blend], default: 360\n"
            "\t--enable-seam optional, enable seam finder in blending area, default: no\n"
            "\t--help        usage\n",
            arg0);
}

static XCamReturn
dump_buffer (SmartPtr<DrmBoBuffer> buffer, char *dump_name)
{
    ImageFileHandle file;

    XCamReturn ret = file.open (dump_name, "wb");
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("open %s failed", dump_name);
        return ret;
    }

    ret = file.write_buf (buffer);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("write buffer to %s failed", dump_name);
        file.close ();
        return ret;
    }

    file.close ();
    XCAM_LOG_INFO ("write buffer to %s done", dump_name);

    return XCAM_RETURN_NO_ERROR;
}

static void
ensure_gpu_buffer_done (SmartPtr<BufferProxy> buf)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            int mem_idx = info.offsets [index] + i * info.strides [index] + line_bytes - 1;
            if (memory[mem_idx] == 1) {
                memory[mem_idx] = 1;
            }
        }
    }
    buf->unmap ();
}

static void
get_fisheye_info (CLFisheyeInfo &info0, CLFisheyeInfo &info1)
{
    info0.center_x = 480.0f;
    info0.center_y = 480.0f;
    info0.wide_angle = 202.8f;
    info0.radius = 480.0f;
    info0.rotate_angle = -90.0f;

    info1.center_x = 1440.0f;
    info1.center_y = 480.0f;
    info1.wide_angle = 202.8f;
    info1.radius = 480.0f;
    info1.rotate_angle = 89.4f;
}

static XCamReturn
fisheye_correction (
    SmartPtr<CLFisheyeHandler> fisheye_handler, const CLFisheyeInfo fisheye_info,
    SmartPtr<DrmBoBuffer> input_buf, SmartPtr<DrmBoBuffer> &output_buf,
    uint32_t output_width, uint32_t output_height,
    char *dump_name, bool need_save_output)
{
    fisheye_handler->set_fisheye_info (fisheye_info);
    fisheye_handler->set_dst_range (max_dst_angle, 180.0f);
    fisheye_handler->set_output_size (output_width, output_height);

    XCamReturn ret = fisheye_handler->execute (input_buf, output_buf);
    if (ret != XCAM_RETURN_NO_ERROR) {
        XCAM_LOG_ERROR ("fisheye handler execute failed");
        return ret;
    }
    XCAM_ASSERT (output_buf.ptr ());

    if (need_save_output) {
        ret = dump_buffer (output_buf, dump_name);
        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_ERROR ("fisheye correction: dump buffer failed");
            return ret;
        }
    }

    return XCAM_RETURN_NO_ERROR;
}

static void
get_stitch_info (
    uint32_t fisheye_width, uint32_t fisheye_height,
    FisheyeCropInfo crop0, FisheyeCropInfo crop1,
    ImageStitchInfo &info0, ImageStitchInfo &info1)
{
    info0.stitch_left.pos_x = crop0.left;
    info0.stitch_left.pos_y = crop0.top;
    info0.stitch_left.width = merge_width0;
    info0.stitch_left.height = fisheye_height - crop0.top - crop0.bottom;
    info0.stitch_right.pos_x = fisheye_width - crop0.right - merge_width1;
    info0.stitch_right.pos_y = crop0.top;
    info0.stitch_right.width = merge_width1;
    info0.stitch_right.height = fisheye_height - crop0.top - crop0.bottom;

    info1.stitch_left.pos_x = crop1.left;
    info1.stitch_left.pos_y = crop1.top;
    info1.stitch_left.width = merge_width1;
    info1.stitch_left.height = fisheye_height - crop1.top - crop1.bottom;
    info1.stitch_right.pos_x = fisheye_width - crop1.right - merge_width0;
    info1.stitch_right.pos_y = crop1.top;
    info1.stitch_right.width = merge_width0;
    info1.stitch_right.height = fisheye_height - crop1.top - crop1.bottom;
}

static XCamReturn
image_360_stitch (
    SmartPtr<CLImage360Stitch> image_360, int index,
    SmartPtr<DrmBoBuffer> input0,
    SmartPtr<DrmBoBuffer> input1,
    SmartPtr<DrmBoBuffer> &output_buf)
{
    SmartPtr<DrmBoBuffer> input;
    if (index % 2 == 0) {
        input1->attach_buffer (input0);
        input = input1;
    } else {
        input0->attach_buffer (input1);
        input = input0;
    }

    return image_360->execute (input, output_buf);
}

static void
set_blend_info (
    SmartPtr<CLBlender> blender,
    uint32_t output_width, uint32_t output_height,
    uint32_t fisheye_width, FisheyeCropInfo crop0, FisheyeCropInfo crop1)
{
    Rect area;
    area.pos_x = 0;
    area.pos_y = crop0.top;
    area.height = output_height;
    area.width = fisheye_width - crop0.right;
    blender->set_input_valid_area (area, 0);
    area.pos_x = crop1.left;
    area.width = fisheye_width - crop1.left;
    blender->set_input_valid_area (area, 1);

    area.pos_x = fisheye_width - crop0.right - merge_width1;
    area.width = merge_width1;
    blender->set_merge_window (area);

    blender->set_input_merge_area (area, 0);
    area.pos_x = crop1.left;
    area.width = merge_width1;
    blender->set_input_merge_area (area, 1);

    blender->set_output_size (output_width, output_height);
}

static XCamReturn
blend_images (
    SmartPtr<CLBlender> blender,
    SmartPtr<DrmBoBuffer> input0,
    SmartPtr<DrmBoBuffer> input1,
    SmartPtr<DrmBoBuffer> &output_buf)
{
    input0->attach_buffer (input1);
    return blender->execute (input0, output_buf);
}

int main (int argc, char *argv[])
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLContext> context;
    SmartPtr<DrmDisplay> display;
    SmartPtr<BufferPool> buf_pool;
    SmartPtr<BufferProxy> read_buf;
    ImageFileHandle file_in, file_out;
    SmartPtr<DrmBoBuffer> input_buf, output_buf;
    VideoBufferInfo input_buf_info, output_buf_info;
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLFisheyeHandler> fisheye;
    CLFisheyeInfo fisheye_info0, fisheye_info1;
    SmartPtr<DrmBoBuffer> fisheye_buf0, fisheye_buf1;
    SmartPtr<CLBlender> blender;
    SmartPtr<CLImage360Stitch> image_360;

    uint32_t input_format = V4L2_PIX_FMT_NV12;
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t output_height = 960;
    uint32_t output_width = output_height * 2;

    int loop = 0;
    bool enable_seam = false;
    bool need_save_output = true;
    ImageStitchMode stitch_mode = IMAGE_STITCH_MODE_360;
    char file_in_name[XCAM_MAX_STR_SIZE], file_out_name[XCAM_MAX_STR_SIZE];

    const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},
        {"output", required_argument, NULL, 'o'},
        {"input-w", required_argument, NULL, 'w'},
        {"input-h", required_argument, NULL, 'h'},
        {"output-w", required_argument, NULL, 'W'},
        {"output-h", required_argument, NULL, 'H'},
        {"loop", required_argument, NULL, 'l'},
        {"save", required_argument, NULL, 's'},
        {"stitch-mode", required_argument, NULL, 't'},
        {"enable-seam", no_argument, NULL, 'S'},
        {"help", no_argument, NULL, 'e'},
        {NULL, 0, NULL, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            strncpy (file_in_name, optarg, XCAM_MAX_STR_SIZE);
            break;
        case 'o':
            strncpy (file_out_name, optarg, XCAM_MAX_STR_SIZE);
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
        case 'l':
            loop = atoi(optarg);
            break;
        case 's':
            need_save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 't':
            if (!strcmp (optarg, "360"))
                stitch_mode = IMAGE_STITCH_MODE_360;
            else if (!strcasecmp (optarg, "blend"))
                stitch_mode = IMAGE_STITCH_MODE_BLEND;
            else {
                XCAM_LOG_ERROR ("incorrect stitching mode");
                return -1;
            }
            break;
        case 'S':
            enable_seam = true;
            break;
        case 'e':
            usage (argv[0]);
            return -1;
        default:
            XCAM_LOG_ERROR ("getopt_long return unknown value:%c", opt);
            usage (argv[0]);
            return -1;
        }
    }

    if (optind < argc || argc < 2) {
        XCAM_LOG_ERROR ("unknow option %s", argv[optind]);
        usage (argv[0]);
        return -1;
    }
    // if ((stitch_mode == IMAGE_STITCH_MODE_360) && (output_width != output_height * 2)) {
    //     XCAM_LOG_ERROR ("incorrect output size, please check");
    //     return -1;
    // }

    printf ("Description----------------\n");
    printf ("input file:\t%s\n", file_in_name);
    printf ("output file:\t%s\n", file_out_name);
    printf ("input width:\t%d\n", input_width);
    printf ("input height:\t%d\n", input_height);
    printf ("output width:\t%d\n", output_width);
    printf ("output height:\t%d\n", output_height);
    printf ("loop count:\t%d\n", loop);
    printf ("save file:\t%s\n", need_save_output ? "true" : "false");
    printf ("stitch mode:\t%s\n", stitch_mode == IMAGE_STITCH_MODE_360 ? "360" : "blend");
    printf ("seam mask:\t%s\n", enable_seam ? "true" : "false");
    printf ("---------------------------\n");

    context = CLDevice::instance ()->get_context ();
    fisheye = create_fisheye_handler (context).dynamic_cast_ptr<CLFisheyeHandler> ();
    XCAM_ASSERT (fisheye.ptr ());
    get_fisheye_info (fisheye_info0, fisheye_info1);
    uint32_t fisheye_output_width = (max_dst_angle / 180.0f) * output_height;
    uint32_t fisheye_output_height = output_height;
    XCAM_LOG_INFO (
        "fisheye correction output size width:%d height:%d",
        fisheye_output_width, fisheye_output_height);

    if (stitch_mode == IMAGE_STITCH_MODE_360) {
        output_width = fisheye_output_width * 2 - merge_width0 - merge_width1
                       - fisheye_crop0.left - fisheye_crop0.right - fisheye_crop1.left - fisheye_crop1.right;
        output_height = fisheye_output_height - fisheye_crop0.top - fisheye_crop0.bottom;
        XCAM_LOG_INFO (
            "stitching output size width:%d height:%d",
            output_width, output_height);
        image_360 = create_image_360_stitch (context, enable_seam).dynamic_cast_ptr<CLImage360Stitch> ();
        XCAM_ASSERT (image_360.ptr ());
        ImageStitchInfo stitch_info0, stitch_info1;
        get_stitch_info (fisheye_output_width, fisheye_output_height,
                         fisheye_crop0, fisheye_crop1, stitch_info0, stitch_info1);
        image_360->set_output_size (output_width, output_height);
        image_360->set_image_overlap (0, stitch_info0.stitch_left, stitch_info0.stitch_right);
        image_360->set_image_overlap (1, stitch_info1.stitch_left, stitch_info1.stitch_right);
    } else {
        output_width = fisheye_output_width * 2 - merge_width1 - fisheye_crop0.right - fisheye_crop1.left;
        output_height = fisheye_output_height - fisheye_crop0.top - fisheye_crop0.bottom;
        XCAM_LOG_INFO (
            "blending output size width:%d height:%d",
            output_width, output_height);
        blender = create_pyramid_blender (context, 2, true, enable_seam).dynamic_cast_ptr<CLBlender> ();
        XCAM_ASSERT (blender.ptr ());
        set_blend_info (blender, output_width, output_height,
                        fisheye_output_width, fisheye_crop0, fisheye_crop1);
    }

    input_buf_info.init (input_format, input_width, input_height);
    output_buf_info.init (input_format, output_width, output_height);
    display = DrmDisplay::instance ();
    buf_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (buf_pool.ptr ());
    buf_pool->set_video_info (input_buf_info);
    if (!buf_pool->reserve (2)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    input_buf = buf_pool->get_buffer (buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_ASSERT (input_buf.ptr ());
    ret = file_in.open (file_in_name, "rb");
    CHECK (ret, "open %s failed", file_in_name);
    read_buf = input_buf;
    ret = file_in.read_buf (read_buf);
    CHECK (ret, "read buffer from %s failed", file_in_name);

    char dump_name[1024];
    int i = 0;
    do {
        snprintf (dump_name, 1023, "fisheye-0-%s.%02d", file_out_name, i);
        ret = fisheye_correction (fisheye, fisheye_info0, input_buf, fisheye_buf0,
                                  fisheye_output_width, fisheye_output_height, dump_name, need_save_output);
        CHECK (ret, "fisheye_correction execute failed");

        snprintf (dump_name, 1023, "fisheye-1-%s.%02d", file_out_name, i);
        ret = fisheye_correction (fisheye, fisheye_info1, input_buf, fisheye_buf1,
                                  fisheye_output_width, fisheye_output_height, dump_name, need_save_output);
        CHECK (ret, "fisheye_correction execute failed");

        if (stitch_mode == IMAGE_STITCH_MODE_360) {
            ret = image_360_stitch (image_360, i, fisheye_buf0, fisheye_buf1, output_buf);
            CHECK (ret, "image_360 stitch execute failed");
        } else {
            ret = blend_images ( blender, fisheye_buf0, fisheye_buf1, output_buf);
            CHECK (ret, "blend_images execute failed");
        }

        if (need_save_output) {
            snprintf (dump_name, 1023, "%s.%02d", file_out_name, i);
            ret = dump_buffer (output_buf, dump_name);
            CHECK (ret, "dump output buffer failed");
        } else {
            ensure_gpu_buffer_done (output_buf);
        }

        FPS_CALCULATION (image_stitching, 100);
        ++i;
    } while (i < loop);

    return 0;
}

