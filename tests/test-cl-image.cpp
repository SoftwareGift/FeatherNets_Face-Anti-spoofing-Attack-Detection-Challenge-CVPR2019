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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 * Author: Wei Zong <wei.zong@intel.com>
 */

#include "test_common.h"
#include "test_inline.h"
#include "image_file_handle.h"
#include "drm_bo_buffer.h"
#include "ocl/cl_device.h"
#include "ocl/cl_context.h"
#include "ocl/cl_demo_handler.h"
#include "ocl/cl_csc_handler.h"
#include "ocl/cl_bayer_pipe_handler.h"
#include "ocl/cl_yuv_pipe_handler.h"
#include "ocl/cl_tonemapping_handler.h"
#include "ocl/cl_retinex_handler.h"
#include "ocl/cl_gauss_handler.h"
#include "ocl/cl_wavelet_denoise_handler.h"
#include "ocl/cl_newwavelet_denoise_handler.h"
#include "ocl/cl_defog_dcp_handler.h"
#include "ocl/cl_3d_denoise_handler.h"
#include "ocl/cl_image_warp_handler.h"
#include "ocl/cl_fisheye_handler.h"

using namespace XCam;

enum TestHandlerType {
    TestHandlerUnknown  = 0,
    TestHandlerDemo,
    TestHandlerColorConversion,
    TestHandlerBayerPipe,
    TestHandlerYuvPipe,
    TestHandlerTonemapping,
    TestHandlerRetinex,
    TestHandlerGauss,
    TestHandlerHatWavelet,
    TestHandlerHaarWavelet,
    TestHandlerDefogDcp,
    TestHandler3DDenoise,
    TestHandlerImageWarp,
    TestHandlerFisheye,
};

enum PsnrType {
    PSNRY = 0,
    PSNRR,
    PSNRG,
    PSNRB,
};

static XCamReturn
calculate_psnr (SmartPtr<DrmBoBuffer> &psnr_cur, SmartPtr<DrmBoBuffer> &psnr_ref, PsnrType psnr_type, float &psnr)
{
    const VideoBufferInfo info = psnr_cur->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *cur_mem = NULL, *ref_mem = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    int8_t interval = 1, index = 0;
    if (PSNRY == psnr_type) {
        interval = 1;
        index = 0;
    } else if (PSNRR == psnr_type) {
        interval = 4;
        index = 0;
    } else if (PSNRG == psnr_type) {
        interval = 4;
        index = 1;
    } else if (PSNRB == psnr_type) {
        interval = 4;
        index = 2;
    }

    cur_mem = psnr_cur->map ();
    ref_mem = psnr_ref->map ();
    if (!cur_mem || !ref_mem) {
        XCAM_LOG_ERROR ("calculate_psnr map buffer failed");
        return XCAM_RETURN_ERROR_MEM;
    }

    uint32_t sum = 0, pos = 0;
    info.get_planar_info (planar, 0);
    for (uint32_t i = 0; i < planar.height; i++) {
        for (uint32_t j = 0; j < planar.width / interval; j++) {
            pos = i * planar.width + j * interval + index;
            sum += (cur_mem [pos] - ref_mem [pos]) * (cur_mem [pos] - ref_mem [pos]);
        }
    }
    float mse = (float) sum / (planar.height * planar.width / interval) + 0.000001f;
    psnr = 10 * log10 (255 * 255 / mse);

    psnr_cur->unmap ();
    psnr_ref->unmap ();

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
            "\t -t type           specify image handler type\n"
            "\t                   select from [demo, blacklevel, defect, demosaic, tonemapping, csc, hdr, wb, denoise,"
            " gamma, snr, bnr, macc, ee, bayerpipe, yuvpipe, retinex, gauss, wavelet-hat, wavelet-haar, dcp, fisheye]\n"
            "\t -f input_format   specify a input format\n"
            "\t -W image_width    specify input image width\n"
            "\t -H image_height   specify input image height\n"
            "\t -g output_format  specify a output format\n"
            "\t                   select from [NV12, BA10, RGBA, RGBA64]\n"
            "\t -i input          specify input file path\n"
            "\t -o output         specify output file path\n"
            "\t -r refer          specify reference file path\n"
            "\t -k binary_kernel  specify binary kernel path\n"
            "\t -p count          specify cl kernel loop count\n"
            "\t -c csc_type       specify csc type, default:rgba2nv12\n"
            "\t                   select from [rgbatonv12, rgbatolab, rgba64torgba, yuyvtorgba, nv12torgba]\n"
            "\t -b                enable bayer-nr, default: disable\n"
            "\t -P                enable psnr calculation, default: disable\n"
            "\t -h                help\n"
            , bin_name);

    printf ("Note:\n"
            "Usage of binary kernel:\n"
            "1. generate binary kernel:\n"
            "   $ test-binary-kernel --src-kernel kernel_demo.cl --bin-kernel kernel_demo.cl.bin"
            " --kernel-name kernel_demo\n"
            "2. execute binary kernel:\n"
            "   $ test-cl-image -t demo -f BA10 -i input.raw -o output.raw -k kernel_demo.cl.bin\n");
}

int main (int argc, char *argv[])
{
    uint32_t input_format = 0;
    uint32_t output_format = V4L2_PIX_FMT_RGBA32;
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint32_t buf_count = 0;
    int32_t kernel_loop_count = 0;
    const char *input_file = NULL, *output_file = NULL, *refer_file = NULL;
    const char *bin_kernel_path = NULL;
    ImageFileHandle input_fp, output_fp, refer_fp;
    const char *bin_name = argv[0];
    TestHandlerType handler_type = TestHandlerUnknown;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLImageHandler> image_handler;
    VideoBufferInfo input_buf_info;
    SmartPtr<CLContext> context;
    SmartPtr<DrmDisplay> display;
    SmartPtr<DrmBoBufferPool> buf_pool;
    int opt = 0;
    CLCscType csc_type = CL_CSC_TYPE_RGBATONV12;
    bool enable_bnr = false;
    bool enable_psnr = false;

    while ((opt =  getopt(argc, argv, "f:W:H:i:o:r:t:k:p:c:g:bPh")) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;
        case 'r':
            refer_file = optarg;
            break;

        case 'f': {
            if (!strcasecmp (optarg, "nv12"))
                input_format = V4L2_PIX_FMT_NV12;
            else if (!strcasecmp (optarg, "ba10"))
                input_format = V4L2_PIX_FMT_SGRBG10;
            else if (! strcasecmp (optarg, "rgba"))
                input_format = V4L2_PIX_FMT_RGBA32;
            else if (! strcasecmp (optarg, "rgba64"))
                input_format = XCAM_PIX_FMT_RGBA64;
            else if (!strcasecmp (optarg, "ba12"))
                input_format = V4L2_PIX_FMT_SGRBG12;
            else
                print_help (bin_name);
            break;
        }
        case 'W': {
            width = atoi (optarg);
            break;
        }
        case 'H': {
            height = atoi (optarg);
            break;
        }
        case 'g': {
            if (!strcasecmp (optarg, "nv12"))
                output_format = V4L2_PIX_FMT_NV12;
            else if (!strcasecmp (optarg, "ba10"))
                output_format = V4L2_PIX_FMT_SGRBG10;
            else if (! strcasecmp (optarg, "rgba"))
                output_format = V4L2_PIX_FMT_RGBA32;
            else if (! strcasecmp (optarg, "rgba64"))
                output_format = XCAM_PIX_FMT_RGBA64;

            else
                print_help (bin_name);
            break;
        }
        case 't': {
            if (!strcasecmp (optarg, "demo"))
                handler_type = TestHandlerDemo;
            else if (!strcasecmp (optarg, "csc"))
                handler_type = TestHandlerColorConversion;
            else if (!strcasecmp (optarg, "bayerpipe"))
                handler_type = TestHandlerBayerPipe;
            else if (!strcasecmp (optarg, "yuvpipe"))
                handler_type = TestHandlerYuvPipe;
            else if (!strcasecmp (optarg, "tonemapping"))
                handler_type = TestHandlerTonemapping;
            else if (!strcasecmp (optarg, "retinex"))
                handler_type = TestHandlerRetinex;
            else if (!strcasecmp (optarg, "gauss"))
                handler_type = TestHandlerGauss;
            else if (!strcasecmp (optarg, "wavelet-hat"))
                handler_type = TestHandlerHatWavelet;
            else if (!strcasecmp (optarg, "wavelet-haar"))
                handler_type = TestHandlerHaarWavelet;
            else if (!strcasecmp (optarg, "dcp"))
                handler_type = TestHandlerDefogDcp;
            else if (!strcasecmp (optarg, "3d-denoise"))
                handler_type = TestHandler3DDenoise;
            else if (!strcasecmp (optarg, "warp"))
                handler_type = TestHandlerImageWarp;
            else if (!strcasecmp (optarg, "fisheye"))
                handler_type = TestHandlerFisheye;
            else
                print_help (bin_name);
            break;
        }
        case 'k':
            bin_kernel_path = optarg;
            break;
        case 'p':
            kernel_loop_count = atoi (optarg);
            XCAM_ASSERT (kernel_loop_count >= 0 && kernel_loop_count < INT32_MAX);
            break;
        case 'c':
            if (!strcasecmp (optarg, "rgbatonv12"))
                csc_type = CL_CSC_TYPE_RGBATONV12;
            else if (!strcasecmp (optarg, "rgbatolab"))
                csc_type = CL_CSC_TYPE_RGBATOLAB;
            else if (!strcasecmp (optarg, "rgba64torgba"))
                csc_type = CL_CSC_TYPE_RGBA64TORGBA;
            else if (!strcasecmp (optarg, "yuyvtorgba"))
                csc_type = CL_CSC_TYPE_YUYVTORGBA;
            else if (!strcasecmp (optarg, "nv12torgba"))
                csc_type = CL_CSC_TYPE_NV12TORGBA;
            else
                print_help (bin_name);
            break;

        case 'b':
            enable_bnr = true;
            break;

        case 'P':
            enable_psnr = true;
            break;

        case 'h':
            print_help (bin_name);
            return 0;

        default:
            print_help (bin_name);
            return -1;
        }
    }

    if (!input_format || !input_file || !output_file || (enable_psnr && !refer_file) || handler_type == TestHandlerUnknown) {
        print_help (bin_name);
        return -1;
    }

    ret = input_fp.open (input_file, "rb");
    CHECK (ret, "open input file(%s) failed", XCAM_STR (input_file));
    ret = output_fp.open (output_file, "wb");
    CHECK (ret, "open output file(%s) failed", XCAM_STR (output_file));
    if (enable_psnr) {
        refer_fp.open (refer_file, "rb");
        CHECK (ret, "open reference file(%s) failed", XCAM_STR (refer_file));
    }

    context = CLDevice::instance ()->get_context ();

    switch (handler_type) {
    case TestHandlerDemo:
        if (!bin_kernel_path)
            image_handler = create_cl_demo_image_handler (context);
        else {
            FileHandle file;
            if (file.open (bin_kernel_path, "r") != XCAM_RETURN_NO_ERROR) {
                XCAM_LOG_ERROR ("open binary kernel failed");
                return -1;
            }

            size_t size;
            if (file.get_file_size (size) != XCAM_RETURN_NO_ERROR) {
                XCAM_LOG_ERROR ("get binary kernel size failed");
                return -1;
            }

            uint8_t *binary = (uint8_t *) xcam_malloc0 (sizeof (uint8_t) * (size));
            XCAM_ASSERT (binary);

            if (file.read_file (binary, size) != XCAM_RETURN_NO_ERROR) {
                XCAM_LOG_ERROR ("read binary kernel failed");
                xcam_free (binary);
                return -1;
            }

            image_handler = create_cl_binary_demo_image_handler (context, binary, size);
            xcam_free (binary);
        }
        break;
    case TestHandlerColorConversion: {
        SmartPtr<CLCscImageHandler> csc_handler;
        XCam3aResultColorMatrix color_matrix;
        xcam_mem_clear (color_matrix);
        double matrix_table[XCAM_COLOR_MATRIX_SIZE] = {0.299, 0.587, 0.114, -0.14713, -0.28886, 0.436, 0.615, -0.51499, -0.10001};
        memcpy (color_matrix.matrix, matrix_table, sizeof(double)*XCAM_COLOR_MATRIX_SIZE);
        image_handler = create_cl_csc_image_handler (context, csc_type);
        csc_handler = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
        XCAM_ASSERT (csc_handler.ptr ());
        csc_handler->set_matrix(color_matrix);
        break;
    }
    case TestHandlerBayerPipe: {
        image_handler = create_cl_bayer_pipe_image_handler (context);
        SmartPtr<CLBayerPipeImageHandler> bayer_pipe = image_handler.dynamic_cast_ptr<CLBayerPipeImageHandler> ();
        XCAM_ASSERT (bayer_pipe.ptr ());
        bayer_pipe->set_output_format (output_format);
        bayer_pipe->enable_denoise (enable_bnr);
        break;
    }
    case TestHandlerYuvPipe: {
        image_handler = create_cl_yuv_pipe_image_handler (context);
        SmartPtr<CLYuvPipeImageHandler> yuv_pipe = image_handler.dynamic_cast_ptr<CLYuvPipeImageHandler> ();
        XCAM_ASSERT (yuv_pipe.ptr ());
        break;
    }
    case TestHandlerTonemapping: {
        image_handler = create_cl_tonemapping_image_handler (context);
        SmartPtr<CLTonemappingImageHandler> tonemapping_pipe = image_handler.dynamic_cast_ptr<CLTonemappingImageHandler> ();
        XCAM_ASSERT (tonemapping_pipe.ptr ());
        break;
    }
    case TestHandlerRetinex: {
        image_handler = create_cl_retinex_image_handler (context);
        SmartPtr<CLRetinexImageHandler> retinex = image_handler.dynamic_cast_ptr<CLRetinexImageHandler> ();
        XCAM_ASSERT (retinex.ptr ());
        break;
    }
    case TestHandlerGauss: {
        image_handler = create_cl_gauss_image_handler (context);
        SmartPtr<CLGaussImageHandler> gauss = image_handler.dynamic_cast_ptr<CLGaussImageHandler> ();
        XCAM_ASSERT (gauss.ptr ());
        break;
    }
    case TestHandlerHatWavelet: {
        image_handler = create_cl_wavelet_denoise_image_handler (context, CL_IMAGE_CHANNEL_UV);
        SmartPtr<CLWaveletDenoiseImageHandler> wavelet = image_handler.dynamic_cast_ptr<CLWaveletDenoiseImageHandler> ();
        XCAM_ASSERT (wavelet.ptr ());
        XCam3aResultWaveletNoiseReduction wavelet_config;
        xcam_mem_clear (wavelet_config);
        wavelet_config.threshold[0] = 0.2;
        wavelet_config.threshold[1] = 0.5;
        wavelet_config.decomposition_levels = 4;
        wavelet_config.analog_gain = 0.001;
        wavelet->set_denoise_config (wavelet_config);
        break;
    }
    case TestHandlerHaarWavelet: {
        image_handler = create_cl_newwavelet_denoise_image_handler (context, CL_IMAGE_CHANNEL_UV | CL_IMAGE_CHANNEL_Y, false);
        SmartPtr<CLNewWaveletDenoiseImageHandler> wavelet = image_handler.dynamic_cast_ptr<CLNewWaveletDenoiseImageHandler> ();
        XCAM_ASSERT (wavelet.ptr ());
        XCam3aResultWaveletNoiseReduction wavelet_config;
        wavelet_config.threshold[0] = 0.2;
        wavelet_config.threshold[1] = 0.5;
        wavelet_config.decomposition_levels = 4;
        wavelet_config.analog_gain = 0.001;
        wavelet->set_denoise_config (wavelet_config);
        break;
    }
    case TestHandlerDefogDcp: {
        image_handler = create_cl_defog_dcp_image_handler (context);
        XCAM_ASSERT (image_handler.ptr ());
        break;
    }
    case TestHandler3DDenoise: {
        uint8_t ref_count = 2;
        image_handler = create_cl_3d_denoise_image_handler (context, CL_IMAGE_CHANNEL_Y | CL_IMAGE_CHANNEL_UV, ref_count);
        SmartPtr<CL3DDenoiseImageHandler> denoise = image_handler.dynamic_cast_ptr<CL3DDenoiseImageHandler> ();
        XCAM_ASSERT (denoise.ptr ());
        XCam3aResultTemporalNoiseReduction denoise_config;
        xcam_mem_clear (denoise_config);
        denoise_config.threshold[0] = 0.05;
        denoise_config.threshold[1] = 0.05;
        denoise_config.gain = 0.6;
        denoise->set_denoise_config (denoise_config);
        break;
    }
    case TestHandlerImageWarp: {
        image_handler = create_cl_image_warp_handler (context);
        SmartPtr<CLImageWarpHandler> warp = image_handler.dynamic_cast_ptr<CLImageWarpHandler> ();
        XCAM_ASSERT (warp.ptr ());
        XCamDVSResult warp_config;
        xcam_mem_clear (warp_config);
        warp_config.frame_id = 1;
        warp_config.frame_width = width;
        warp_config.frame_height = height;

        float theta = -10.0f;
        float phi = 10.0f;

        float shift_x = -0.2f * width;
        float shift_y = 0.2f * height;
        float scale_x = 2.0f;
        float scale_y = 0.5f;
        float shear_x = tan(theta * 3.1415926 / 180.0f);
        float shear_y = tan(phi * 3.1415926 / 180.0f);
        float project_x = 2.0f / width;
        float project_y = -1.0f / height;

        warp_config.proj_mat[0] = scale_x;
        warp_config.proj_mat[1] = shear_x;
        warp_config.proj_mat[2] = shift_x;
        warp_config.proj_mat[3] = shear_y;
        warp_config.proj_mat[4] = scale_y;
        warp_config.proj_mat[5] = shift_y;
        warp_config.proj_mat[6] = project_x;
        warp_config.proj_mat[7] = project_y;
        warp_config.proj_mat[8] = 1.0f;

        warp->set_warp_config (warp_config);
        break;
    }
    case TestHandlerFisheye: {
        image_handler = create_fisheye_handler (context);
        SmartPtr<CLFisheyeHandler> fisheye = image_handler.dynamic_cast_ptr<CLFisheyeHandler> ();
        XCAM_ASSERT (fisheye.ptr ());
        CLFisheyeInfo fisheye_info;
        //fisheye0 {480.0f, 480.0f, 190.0f, 480.0f, -90.0f},
        //fisheye1 {1440.0f, 480.0f, 190.0f, 480.0f, 90.0f}
        fisheye_info.center_x = 480.0f;
        fisheye_info.center_y = 480.0f;
        fisheye_info.wide_angle = 190.0f;
        fisheye_info.radius = 480.0f;
        fisheye_info.rotate_angle = -90.0f;
        fisheye->set_fisheye_info (fisheye_info);
        fisheye->set_dst_range (210.0f, 180.0f);
        fisheye->set_output_size (1120, 960);
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

    input_buf_info.init (input_format, width, height);
    display = DrmDisplay::instance ();
    buf_pool = new DrmBoBufferPool (display);
    XCAM_ASSERT (buf_pool.ptr ());
    buf_pool->set_swap_flags (
        SwappedBuffer::SwapY | SwappedBuffer::SwapUV, SwappedBuffer::OrderY0Y1 | SwappedBuffer::OrderUV0UV1);
    buf_pool->set_video_info (input_buf_info);
    if (!buf_pool->reserve (6)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    SmartPtr<DrmBoBuffer> psnr_cur, psnr_ref;
    while (true) {
        SmartPtr<DrmBoBuffer> input_buf, output_buf;
        SmartPtr<VideoBuffer> tmp_buf = buf_pool->get_buffer (buf_pool);
        input_buf = tmp_buf.dynamic_cast_ptr<DrmBoBuffer> ();

        XCAM_ASSERT (input_buf.ptr ());
        ret = input_fp.read_buf (input_buf);
        if (ret == XCAM_RETURN_BYPASS)
            break;
        if (ret == XCAM_RETURN_ERROR_FILE) {
            XCAM_LOG_ERROR ("read buffer from %s failed", XCAM_STR (input_file));
            return -1;
        }

        if (kernel_loop_count != 0)
        {
            kernel_loop (image_handler, input_buf, output_buf, kernel_loop_count);
            CHECK (ret, "execute kernels failed");
            return 0;
        }

        ret = image_handler->execute (input_buf, output_buf);
        CHECK_EXP ((ret == XCAM_RETURN_NO_ERROR || ret == XCAM_RETURN_BYPASS), "execute kernels failed");
        if (ret == XCAM_RETURN_BYPASS)
            continue;
        context->finish ();
        XCAM_ASSERT (output_buf.ptr ());
        ret = output_fp.write_buf (output_buf);
        CHECK (ret, "write buffer to %s failed", XCAM_STR (output_file));
        psnr_cur = output_buf;

        ++buf_count;
    }

    XCAM_LOG_INFO ("processed %d buffers successfully", buf_count);

    if (enable_psnr) {
        buf_pool = new DrmBoBufferPool (display);
        XCAM_ASSERT (buf_pool.ptr ());
        buf_pool->set_video_info (input_buf_info);
        if (!buf_pool->reserve (6)) {
            XCAM_LOG_ERROR ("init buffer pool failed");
            return -1;
        }

        SmartPtr<VideoBuffer> tmp_buf = buf_pool->get_buffer (buf_pool);
        psnr_ref = tmp_buf.dynamic_cast_ptr<DrmBoBuffer> ();
        XCAM_ASSERT (psnr_ref.ptr ());

        ret = refer_fp.read_buf (psnr_ref);
        CHECK (ret, "read buffer from %s failed", refer_file);

        float psnr = 0.0f;
        ret = calculate_psnr (psnr_cur, psnr_ref, PSNRY, psnr);
        CHECK (ret, "calculate PSNR_Y failed");
        XCAM_LOG_INFO ("PSNR_Y: %.2f", psnr);

        image_handler = create_cl_csc_image_handler (context, CL_CSC_TYPE_NV12TORGBA);
        XCAM_ASSERT (image_handler.ptr ());

        SmartPtr<DrmBoBuffer> psnr_cur_output, psnr_ref_output;
        ret = image_handler->execute (psnr_cur, psnr_cur_output);
        CHECK (ret, "execute kernels failed");
        XCAM_ASSERT (psnr_cur_output.ptr ());

        ret = image_handler->execute (psnr_ref, psnr_ref_output);
        CHECK (ret, "execute kernels failed");
        XCAM_ASSERT (psnr_ref_output.ptr ());

        ret = calculate_psnr (psnr_cur_output, psnr_ref_output, PSNRR, psnr);
        CHECK (ret, "calculate PSNR_R failed");
        XCAM_LOG_INFO ("PSNR_R: %.2f", psnr);

        ret = calculate_psnr (psnr_cur_output, psnr_ref_output, PSNRG, psnr);
        CHECK (ret, "calculate PSNR_G failed");
        XCAM_LOG_INFO ("PSNR_G: %.2f", psnr);

        ret = calculate_psnr (psnr_cur_output, psnr_ref_output, PSNRB, psnr);
        CHECK (ret, "calculate PSNR_B failed");
        XCAM_LOG_INFO ("PSNR_B: %.2f", psnr);
    }

    return 0;
}
