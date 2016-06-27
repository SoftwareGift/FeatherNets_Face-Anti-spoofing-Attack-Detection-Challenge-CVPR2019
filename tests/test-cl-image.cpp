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
#include "cl_snr_handler.h"
#include "cl_macc_handler.h"
#include "cl_ee_handler.h"
#include "cl_dpc_handler.h"
#include "cl_bnr_handler.h"
#include "cl_bayer_pipe_handler.h"
#include "cl_yuv_pipe_handler.h"
#include "cl_tonemapping_handler.h"
#include "cl_retinex_handler.h"
#include "cl_gauss_handler.h"
#include "cl_wavelet_denoise_handler.h"
#include "cl_newwavelet_denoise_handler.h"
#include "cl_defog_dcp_handler.h"

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
    TestHandlerSimpleNoiseReduction,
    TestHandlerBayerNoiseReduction,
    TestHandlerMacc,
    TestHandlerEe,
    TestHandlerBayerPipe,
    TestHandlerYuvPipe,
    TestHandlerTonemapping,
    TestHandlerRetinex,
    TestHandlerGauss,
    TestHandlerHatWavelet,
    TestHandlerHaarWavelet,
    TestHandlerDefogDcp,
};

enum PsnrType {
    PSNRY = 0,
    PSNRR,
    PSNRG,
    PSNRB,
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
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fread (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, file.fp) != line_bytes) {
                if (feof (file.fp))
                    ret = XCAM_RETURN_BYPASS;
                else {
                    XCAM_LOG_ERROR ("read file failed, size doesn't match");
                    ret = XCAM_RETURN_ERROR_FILE;
                }
            }
        }
    }
    buf->unmap ();
    return ret;
}

static XCamReturn
write_buf (SmartPtr<DrmBoBuffer> &buf, TestFileHandle &file)
{
    const VideoBufferInfo info = buf->get_video_info ();
    VideoBufferPlanarInfo planar;
    uint8_t *memory = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    memory = buf->map ();
    for (uint32_t index = 0; index < info.components; index++) {
        info.get_planar_info (planar, index);
        uint32_t line_bytes = planar.width * planar.pixel_bytes;

        for (uint32_t i = 0; i < planar.height; i++) {
            if (fwrite (memory + info.offsets [index] + i * info.strides [index], 1, line_bytes, file.fp) != line_bytes) {
                XCAM_LOG_ERROR ("read file failed, size doesn't match");
                ret = XCAM_RETURN_ERROR_FILE;
            }
        }
    }
    buf->unmap ();
    return ret;
}

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
            "\t -t type      specify image handler type\n"
            "\t              select from [demo, blacklevel, defect, demosaic, tonemapping, csc, hdr, wb, denoise,"
            " gamma, snr, bnr, macc, ee, bayerpipe, yuvpipe, retinex, gauss, wavelet-hat, wavelet-haar, dcp]\n"
            "\t -f input_format    specify a input format\n"
            "\t -W image width     specify input image width\n"
            "\t -H image height    specify input image height\n"
            "\t -g output_format   specify a output format\n"
            "\t              select from [NV12, BA10, RGBA, RGBA64]\n"
            "\t -i input     specify input file path\n"
            "\t -o output    specify output file path\n"
            "\t -r refer     specify reference file path\n"
            "\t -p count     specify cl kernel loop count\n"
            "\t -c csc_type  specify csc type, default:rgba2nv12\n"
            "\t              select from [rgbatonv12, rgbatolab, rgba64torgba, yuyvtorgba, nv12torgba]\n"
            "\t -d hdr_type  specify hdr type, default:rgb\n"
            "\t              select from [rgb, lab]\n"
            "\t -b           enable bayer-nr, default: disable\n"
            "\t -P           enable psnr calculation, default: disable\n"
            "\t -h           help\n"
            , bin_name);
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
    TestFileHandle input_fp, output_fp, refer_fp;
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
    CLHdrType hdr_type = CL_HDR_TYPE_RGB;
    bool enable_bnr = false;
    bool enable_psnr = false;

    while ((opt =  getopt(argc, argv, "f:W:H:i:o:r:t:p:c:d:g:bPh")) != -1) {
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
            else if (!strcasecmp (optarg, "snr"))
                handler_type = TestHandlerSimpleNoiseReduction;
            else if (!strcasecmp (optarg, "bnr"))
                handler_type = TestHandlerBayerNoiseReduction;
            else if (!strcasecmp (optarg, "macc"))
                handler_type = TestHandlerMacc;
            else if (!strcasecmp (optarg, "ee"))
                handler_type = TestHandlerEe;
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
            else
                print_help (bin_name);
            break;
        }
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
        case 'd':
            if (!strcasecmp (optarg, "rgb"))
                hdr_type = CL_HDR_TYPE_RGB;
            else if (!strcasecmp (optarg, "lab"))
                hdr_type = CL_HDR_TYPE_LAB;
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

    input_fp.fp = fopen (input_file, "rb");
    output_fp.fp = fopen (output_file, "wb");
    if (enable_psnr) {
        refer_fp.fp = fopen (refer_file, "rb");
    }
    if (!input_fp.fp || !output_fp.fp || (enable_psnr && !refer_fp.fp)) {
        XCAM_LOG_ERROR ("open input/output file failed");
        return -1;
    }

    context = CLDevice::instance ()->get_context ();

    switch (handler_type) {
    case TestHandlerDemo:
        image_handler = create_cl_demo_image_handler (context);
        break;
    case TestHandlerBlackLevel: {
        XCam3aResultBlackLevel blc;
        xcam_mem_clear (blc);
        blc.r_level = 0.05;
        blc.gr_level = 0.05;
        blc.gb_level = 0.05;
        blc.b_level = 0.05;
        image_handler = create_cl_blc_image_handler (context);
        SmartPtr<CLBlcImageHandler> blc_handler;
        blc_handler = image_handler.dynamic_cast_ptr<CLBlcImageHandler> ();
        XCAM_ASSERT (blc_handler.ptr ());
        blc_handler->set_blc_config (blc);
        break;
    }
    case TestHandlerDefect:  {
        XCam3aResultDefectPixel dpc;
        xcam_mem_clear (dpc);
        dpc.r_threshold = 0.125;
        dpc.gr_threshold = 0.125;
        dpc.gb_threshold = 0.125;
        dpc.b_threshold = 0.125;
        image_handler = create_cl_dpc_image_handler (context);
        SmartPtr<CLDpcImageHandler> dpc_handler;
        dpc_handler = image_handler.dynamic_cast_ptr<CLDpcImageHandler> ();
        XCAM_ASSERT (dpc_handler.ptr ());
        dpc_handler->set_dpc_config (dpc);
        break;
    }
    break;
    case TestHandlerDemosaic: {
        SmartPtr<CLBayer2RGBImageHandler> ba2rgb_handler;
        image_handler = create_cl_demosaic_image_handler (context);
        ba2rgb_handler = image_handler.dynamic_cast_ptr<CLBayer2RGBImageHandler> ();
        XCAM_ASSERT (ba2rgb_handler.ptr ());
        ba2rgb_handler->set_output_format (output_format);
        break;
    }
    case TestHandlerColorConversion: {
        SmartPtr<CLCscImageHandler> csc_handler;
        XCam3aResultColorMatrix color_matrix;
        xcam_mem_clear (color_matrix);
        double matrix_table[XCAM_COLOR_MATRIX_SIZE] = {0.299, 0.587, 0.114, -0.14713, -0.28886, 0.436, 0.615, -0.51499, -0.10001};
        memcpy (color_matrix.matrix, matrix_table, sizeof(double)*XCAM_COLOR_MATRIX_SIZE);
        image_handler = create_cl_csc_image_handler (context, csc_type);
        csc_handler = image_handler.dynamic_cast_ptr<CLCscImageHandler> ();
        XCAM_ASSERT (csc_handler.ptr ());
        csc_handler->set_rgbtoyuv_matrix(color_matrix);
        break;
    }
    case TestHandlerHDR:
        image_handler = create_cl_hdr_image_handler (context, hdr_type);
        break;
    case TestHandlerDenoise:
        image_handler = create_cl_denoise_image_handler (context);
        break;
    case TestHandlerSimpleNoiseReduction:
        image_handler = create_cl_snr_image_handler (context);
        break;
    case TestHandlerWhiteBalance: {
        XCam3aResultWhiteBalance wb;
        xcam_mem_clear (wb);
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
        xcam_mem_clear (gamma_table);
        for(int i = 0; i < XCAM_GAMMA_TABLE_SIZE; ++i)
            gamma_table.table[i] = (double)(pow(i / 255.0, 1 / 2.2) * 255.0);
        SmartPtr<CLGammaImageHandler> gamma_handler;
        image_handler = create_cl_gamma_image_handler (context);
        gamma_handler = image_handler.dynamic_cast_ptr<CLGammaImageHandler> ();
        XCAM_ASSERT (gamma_handler.ptr ());
        gamma_handler->set_gamma_table (gamma_table);
        break;
    }
    case TestHandlerBayerNoiseReduction: {
        XCam3aResultBayerNoiseReduction bnr;
        xcam_mem_clear (bnr);
        bnr.bnr_gain = 0.2;
        bnr.direction = 0.01;
        image_handler = create_cl_bnr_image_handler (context);
        SmartPtr<CLBnrImageHandler> bnr_handler;
        bnr_handler = image_handler.dynamic_cast_ptr<CLBnrImageHandler> ();
        XCAM_ASSERT (bnr_handler.ptr ());
        bnr_handler->set_bnr_config (bnr);
        break;
    }
    case TestHandlerMacc:
        image_handler = create_cl_macc_image_handler (context);
        break;
    case TestHandlerEe: {
        XCam3aResultEdgeEnhancement ee;
        XCam3aResultNoiseReduction nr;
        xcam_mem_clear (ee);
        xcam_mem_clear (nr);
        ee.gain = 2.0;
        ee.threshold = 150.0;
        nr.gain = 0.1;
        SmartPtr<CLEeImageHandler> ee_handler;
        image_handler = create_cl_ee_image_handler (context);
        ee_handler = image_handler.dynamic_cast_ptr<CLEeImageHandler> ();
        XCAM_ASSERT (ee_handler.ptr ());
        ee_handler->set_ee_config_ee (ee);
        ee_handler->set_ee_config_nr (nr);
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

        SmartPtr<BufferProxy> tmp_buf = buf_pool->get_buffer (buf_pool);
        psnr_ref = tmp_buf.dynamic_cast_ptr<DrmBoBuffer> ();
        XCAM_ASSERT (psnr_ref.ptr ());

        ret = read_buf (psnr_ref, refer_fp);
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
