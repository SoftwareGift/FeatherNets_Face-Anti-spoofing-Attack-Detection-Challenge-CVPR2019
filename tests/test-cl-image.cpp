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
#include "cl_wb_handler.h"
#include "cl_denoise_handler.h"
#include "cl_gamma_handler.h"

using namespace XCam;

enum TestHandlerType {
    TestHandlerUnknown  = 0,
    TestHandlerDemo,
    TestHandlerBlackLevel,
    TestHandlerDefect,
    TestHandlerDemosaic,
    TestHandlerColorConversion,
    TestHandlerHDR,
    TestHandlerWhiteBalance,
    TestHandlerDenoise,
    TestHandlerGamma,
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
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    for (uint32_t i = 0; i < kernel_loop_count; i++) {
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
            "\t              select from [demo, blacklevel, defect, demosaic, csc, hdr, wb, denoise, gamma]\n"
            "\t -f format    specify a format\n"
            "\t              select from [NV12, BA10, RGBA]\n"
            "\t -i input     specify input file path\n"
            "\t -o output    specify output file path\n"
            "\t -p count     specify cl kernel loop count\n"
            "\t -c csc_type  specify csc type, default:rgba2nv12\n"
            "\t              select from [rgba2nv12, rgba2lab]\n"
            "\t -d hdr_type  specify hdr type, default:rgb\n"
            "\t              select from [rgb, lab]\n"
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
    SmartPtr<CLContext> context;
    SmartPtr<DrmDisplay> display;
    SmartPtr<BufferPool> buf_pool;
    int opt = 0;
    CLCscType csc_type = CL_CSC_TYPE_RGBATONV12;
    CLHdrType hdr_type = CL_HDR_TYPE_RGB;

    while ((opt =  getopt(argc, argv, "f:i:o:t:p:c:d:h")) != -1) {
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
                format = V4L2_PIX_FMT_RGBA32;

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
            else if (!strcasecmp (optarg, "wb"))
                handler_type = TestHandlerWhiteBalance;
            else if (!strcasecmp (optarg, "denoise"))
                handler_type = TestHandlerDenoise;
            else if (!strcasecmp (optarg, "gamma"))
                handler_type = TestHandlerGamma;
            else
                print_help (bin_name);
            break;
        }
        case 'p':
            kernel_loop_count = atoi (optarg);
            break;
        case 'c':
            if (!strcasecmp (optarg, "rgba2nv12"))
                csc_type = CL_CSC_TYPE_RGBATONV12;
            else if (!strcasecmp (optarg, "rgba2lab"))
                csc_type = CL_CSC_TYPE_RGBATOLAB;
            else
                print_help (bin_name);
            break;
        case 'd':
            if (!strcasecmp (optarg, "rgb"))
                hdr_type = CL_HDR_TYPE_RGB;
            else if (!strcasecmp (optarg, "lab"))
                hdr_type = CL_HDR_TYPE_LAB;
            else
                print_help (bin_name);
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
        ba2rgb_handler->set_output_format (V4L2_PIX_FMT_BGR32);
        break;
    }
    case TestHandlerColorConversion: {
        image_handler = create_cl_csc_image_handler (context, csc_type);
        break;
    }
    case TestHandlerHDR:
        image_handler = create_cl_hdr_image_handler (context, hdr_type);
        break;
    case TestHandlerDenoise:
        image_handler = create_cl_denoise_image_handler (context);
        break;
    case TestHandlerWhiteBalance: {
        XCam3aResultWhiteBalance wb;
        wb.r_gain = 1.0;
        wb.gr_gain = 1.0;
        wb.gb_gain = 1.0;
        wb.b_gain = 1.0;
        SmartPtr<CLWbImageHandler> wb_handler;
        image_handler = create_cl_wb_image_handler (context);
        wb_handler = image_handler.dynamic_cast_ptr<CLWbImageHandler> ();
        XCAM_ASSERT (wb_handler.ptr ());
        wb_handler->set_wb_config (wb);
        break;
    }
    case TestHandlerGamma: {
        XCam3aResultGammaTable gamma_table;
        for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; ++i)
            gamma_table.table[i] = (double)(pow(i / 255.0, 1 / 2.2) * 255.0);
        SmartPtr<CLGammaImageHandler> gamma_handler;
        image_handler = create_cl_gamma_image_handler (context);
        gamma_handler = image_handler.dynamic_cast_ptr<CLGammaImageHandler> ();
        XCAM_ASSERT (gamma_handler.ptr ());
        gamma_handler->set_gamma_table (gamma_table);
        break;
    }

    default:
        XCAM_LOG_ERROR ("unsupported image handler type:%d", handler_type);
        return -1;
    }
    if (!image_handler.ptr ()) {
        XCAM_LOG_ERROR ("create image_handler failed");
        return -1;
    }

    input_buf_info.init (format, 1920, 1080);
    display = DrmDisplay::instance ();
    buf_pool = new DrmBoBufferPool (display);
    buf_pool->set_video_info (input_buf_info);
    if (!buf_pool->reserve (6)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    while (!feof (input_fp.fp)) {
        SmartPtr<DrmBoBuffer> input_buf, output_buf;
        SmartPtr<BufferProxy> tmp_buf = buf_pool->get_buffer (buf_pool);
        input_buf = tmp_buf.dynamic_cast_ptr<DrmBoBuffer> ();

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
