/*
 * test_cl_image.cpp - test cl image
 *
 *  Copyright (c) 2014-2015 Intel Corporation
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
#include "cl_device.h"
#include "cl_context.h"
#include "cl_demo_handler.h"
#include "cl_hdr_handler.h"
#include "cl_blc_handler.h"
#include "drm_bo_buffer.h"
#include "cl_demosaic_handler.h"
#include "cl_csc_handler.h"

using namespace XCam;

enum TestHandlerType {
    TestHandlerUnknown  = 0,
    TestHandlerDemo,
    TestHandlerBlackLevel,
    TestHandlerDefect,
    TestHandlerDemosaic,
    TestHandlerColorConversion,
    TestHandlerHDR,
};

struct TestFileHandle {
    FILE *fp;
    TestFileHandle ()
        : fp (NULL)
    {}
    ~TestFileHandle ()
    {
        if (fp)
            fclose (fp);
    }
};

static XCamReturn
read_buf (SmartPtr<DrmBoBuffer> &buf, TestFileHandle &file)
{
    const VideoBufferInfo info = buf->get_video_info ();
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    memory = buf->map ();
    if (fread (memory, 1, info.size, file.fp) != info.size) {
        if (feof (file.fp))
            ret = XCAM_RETURN_BYPASS;
        else {
            XCAM_LOG_ERROR ("read file failed, size doesn't match");
            ret = XCAM_RETURN_ERROR_FILE;
        }
    }
    buf->unmap ();
    return ret;
}

static XCamReturn
write_buf (SmartPtr<DrmBoBuffer> &buf, TestFileHandle &file)
{
    const VideoBufferInfo info = buf->get_video_info ();
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    memory = buf->map ();
    if (fwrite (memory, 1, info.size, file.fp) != info.size) {
        XCAM_LOG_ERROR ("read file failed, size doesn't match");
        ret = XCAM_RETURN_ERROR_FILE;
    }
    buf->unmap ();
    return ret;
}

static XCamReturn
kernel_loop(SmartPtr<CLImageHandler> &image_handler, SmartPtr<DrmBoBuffer> &input_buf, SmartPtr<DrmBoBuffer> &output_buf, uint32_t kernel_loop_count)
{
    int i;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    for (i = 0; i < kernel_loop_count; i++) {
        PROFILING_START(cl_kernel);
        ret = image_handler->execute (input_buf, output_buf);
        PROFILING_END(cl_kernel, kernel_loop_count)
    }
    return ret;
}

static void
print_help (const char *bin_name)
{
    printf ("Usage: %s [-f format] -i input -o output\n"
            "\t -t type      specify image handler type\n"
            "\t              select from [demo, blacklevel, defect, demosaic, csc, hdr]\n"
            "\t -f format    specify a format\n"
            "\t              select from [NV12, BA10]\n"
            "\t -i input     specify input file path\n"
            "\t -o output    specify output file path\n"
            "\t -p count     specify cl kernel loop count\n"
            "\t -h           help\n"
            , bin_name);
}

int main (int argc, char *argv[])
{
    uint32_t format = 0;
    uint32_t buf_count = 0;
    uint32_t kernel_loop_count = 0;
    const char *input_file = NULL, *output_file = NULL;
    TestFileHandle input_fp, output_fp;
    const char *bin_name = argv[0];
    TestHandlerType handler_type = TestHandlerUnknown;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLImageHandler> image_handler;
    VideoBufferInfo input_buf_info;
    SmartPtr<DrmBoBuffer> input_buf, output_buf;
    SmartPtr<CLContext> context;
    SmartPtr<DrmDisplay> display;
    SmartPtr<DrmBoBufferPool> buf_pool;
    int opt = 0;

    while ((opt =  getopt(argc, argv, "f:i:o:t:p:h")) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;

        case 'f': {
            if (!strcasecmp (optarg, "nv12"))
                format = V4L2_PIX_FMT_NV12;
            else if (!strcasecmp (optarg, "ba10"))
                format = V4L2_PIX_FMT_SGRBG10;
            else if (! strcasecmp (optarg, "rgba"))
                format = V4L2_PIX_FMT_RGB32;
            else
                print_help (bin_name);
            break;
        }
        case 't': {
            if (!strcasecmp (optarg, "demo"))
                handler_type = TestHandlerDemo;
            else if (!strcasecmp (optarg, "blacklevel"))
                handler_type = TestHandlerBlackLevel;
            else if (!strcasecmp (optarg, "defect"))
                handler_type = TestHandlerDefect;
            else if (!strcasecmp (optarg, "demosaic"))
                handler_type = TestHandlerDemosaic;
            else if (!strcasecmp (optarg, "csc"))
                handler_type = TestHandlerColorConversion;
            else if (!strcasecmp (optarg, "hdr"))
                handler_type = TestHandlerHDR;
            else
                print_help (bin_name);
            break;
        }
        case 'p':
            kernel_loop_count = atoi (optarg);
            break;
        case 'h':
            print_help (bin_name);
            return 0;

        default:
            print_help (bin_name);
            return -1;
        }
    }

    if (!format || !input_file || !output_file || handler_type == TestHandlerUnknown) {
        print_help (bin_name);
        return -1;
    }

    input_fp.fp = fopen (input_file, "rb");
    output_fp.fp = fopen (output_file, "wb");
    if (!input_fp.fp || !output_fp.fp) {
        XCAM_LOG_ERROR ("open input/output file failed");
        return -1;
    }

    context = CLDevice::instance ()->get_context ();

    switch (handler_type) {
    case TestHandlerDemo:
        image_handler = create_cl_demo_image_handler (context);
        break;
    case TestHandlerBlackLevel:
        image_handler = create_cl_blc_image_handler (context);
        break;
    case TestHandlerDefect:
        break;
    case TestHandlerDemosaic: {
        SmartPtr<CLBayer2RGBImageHandler> ba2rgb_handler;
        image_handler = create_cl_demosaic_image_handler (context);
        ba2rgb_handler = image_handler.dynamic_cast_ptr<CLBayer2RGBImageHandler> ();
        XCAM_ASSERT (ba2rgb_handler.ptr ());
        ba2rgb_handler->set_output_format (V4L2_PIX_FMT_RGB32);
        break;
    }
    case TestHandlerColorConversion: {
        SmartPtr<CLRgba2Nv12ImageHandler> rgba2nv12_handler;
        image_handler = create_cl_csc_image_handler (context);
        rgba2nv12_handler = image_handler.dynamic_cast_ptr<CLRgba2Nv12ImageHandler> ();
        XCAM_ASSERT (rgba2nv12_handler.ptr ());
        rgba2nv12_handler->set_output_format (V4L2_PIX_FMT_NV12);
        break;
    }
    case TestHandlerHDR:
        image_handler = create_cl_hdr_image_handler (context);
        break;

    default:
        XCAM_LOG_ERROR ("unsupported image handler type:%d", handler_type);
        return -1;
    }
    if (!image_handler.ptr ()) {
        XCAM_LOG_ERROR ("create image_handler failed");
        return -1;
    }
    input_buf_info.format = format;
    input_buf_info.color_bits = 8;
    input_buf_info.width = 1920;
    input_buf_info.height = 1080;
    switch (format) {
    case V4L2_PIX_FMT_NV12:
        input_buf_info.color_bits = 8;
        input_buf_info.components = 2;
        input_buf_info.strides[0] = input_buf_info.width;
        input_buf_info.strides[1] = input_buf_info.strides[0];
        input_buf_info.offsets[0] = 0;
        input_buf_info.offsets[1] = input_buf_info.strides[0] * input_buf_info.height;
        input_buf_info.size = input_buf_info.strides[0] * input_buf_info.height * 3 / 2;
        break;
    case V4L2_PIX_FMT_SGRBG10:
        input_buf_info.color_bits = 10;
        input_buf_info.components = 1;
        input_buf_info.strides[0] = input_buf_info.width * 2;
        input_buf_info.offsets[0] = 0;
        input_buf_info.size = input_buf_info.strides[0] * input_buf_info.height;
        break;
    case V4L2_PIX_FMT_RGB32:
        input_buf_info.color_bits = 8;
        input_buf_info.components = 1;
        input_buf_info.strides[0] = input_buf_info.width * 4;
        input_buf_info.offsets[0] = 0;
        input_buf_info.size = input_buf_info.strides[0] * input_buf_info.height;
        break;
    }

    display = DrmDisplay::instance ();
    buf_pool = new DrmBoBufferPool (display);
    buf_pool->set_buffer_info (input_buf_info);
    if (!buf_pool->init (6)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    while (!feof (input_fp.fp)) {
        input_buf = buf_pool->get_buffer (buf_pool);
        XCAM_ASSERT (input_buf.ptr ());
        ret = read_buf (input_buf, input_fp);
        if (ret == XCAM_RETURN_BYPASS)
            break;
        CHECK (ret, "read buffer from %s failed", input_file);

        if (kernel_loop_count != 0)
        {
            kernel_loop (image_handler, input_buf, output_buf, kernel_loop_count);
            CHECK (ret, "execute kernels failed");
            return 0;
        }

        ret = image_handler->execute (input_buf, output_buf);
        CHECK (ret, "execute kernels failed");
        XCAM_ASSERT (output_buf.ptr ());

        ret = write_buf (output_buf, output_fp);
        CHECK (ret, "read buffer from %s failed", output_file);

        ++buf_count;
    }
    XCAM_LOG_INFO ("processed %d buffers successfully", buf_count);
    return 0;
}
