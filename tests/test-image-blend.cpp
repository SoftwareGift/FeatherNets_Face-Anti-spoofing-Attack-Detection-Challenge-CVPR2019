/*
 * test-image-blend.cpp - test cl image
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "test_common.h"
#include "test_inline.h"
#include <unistd.h>
#include <getopt.h>
#include "ocl/cl_device.h"
#include "ocl/cl_context.h"
#include "ocl/cl_blender.h"
#include "image_file_handle.h"
#include "ocl/cl_geo_map_handler.h"
#include "drm_display.h"
#include "dma_video_buffer.h"

using namespace XCam;

#define ENABLE_DMA_TEST 1

static uint32_t input_format = V4L2_PIX_FMT_NV12;
//static uint32_t output_format = V4L2_PIX_FMT_NV12;
static uint32_t input_width0 = 1280, input_width1 = 1280;
static uint32_t input_height = 960;
static uint32_t output_width = 1920;
static uint32_t output_height;
static bool need_save_output = true;
static bool enable_geo = false;
static bool enable_seam = false;

static int loop = 0;
static uint32_t map_width = 51, map_height = 43;
static const char *map0 = "fisheye0.csv";
static const char *map1 = "fisheye1.csv";
static char file_in0_name[XCAM_MAX_STR_SIZE], file_in1_name[XCAM_MAX_STR_SIZE], file_out_name[XCAM_MAX_STR_SIZE];

static int read_map_data (const char* file, GeoPos *map, int width, int height);

static void
usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input0 file --input1 file --output file"
            " [--input-w0 width] [--input-w1 width] [--input-h height] [--output-w width] \n"
            "\t--input0, first image(NV12)\n"
            "\t--input1, second image(NV12)\n"
            "\t--output, output image(NV12) PREFIX\n"
            "\t--input-w0, optional, input width; default:1280\n"
            "\t--input-w1, optional, input width; default:1280\n"
            "\t--input-h,  optional, input height; default:960\n"
            "\t--output-w, optional, output width; default:1920, output height is same as input height.\n"
            "\t--loop,     optional, how many loops need to run for performance test, default 0; \n"
            "\t--save,     optional, save file or not, default true; select from [true/false]\n"
            "\t--enable-geo,  optional, enable geo map image frist. default: no\n"
            "\t--enable-seam, optional, enable seam finder in blending area. default: no\n"
            "\t--help,     usage\n",
            arg0);
}

static int
geo_correct_image (
    SmartPtr<CLGeoMapHandler> geo_map_handler, SmartPtr<DrmBoBuffer> &in_out,
    GeoPos *geo_map0, uint32_t map_width, uint32_t map_height,
    char *file_name, bool need_save_output)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<DrmBoBuffer> geo_out;
    geo_map_handler->set_map_data (geo_map0, map_width, map_height);
    ret = geo_map_handler->execute (in_out, geo_out);
    CHECK (ret, "geo map handler execute inpu0 failed");
    XCAM_ASSERT (geo_out.ptr ());
    in_out = geo_out;

    if (need_save_output) {
        char gdc_dump_name[1024];
        snprintf (gdc_dump_name, 1024, "gdc-%s", file_name);
        ImageFileHandle file_out;
        file_out.open (gdc_dump_name, "wb");
        file_out.write_buf (geo_out);
        file_out.close ();
        printf ("write gdc output buffer to: %s done\n", gdc_dump_name);
    }
    return 0;
}

static SmartPtr<DrmBoBuffer>
dma_buf_to_xcam_buf (
    SmartPtr<DrmDisplay> display, int dma_fd,
    uint32_t width, uint32_t height, uint32_t size,
    uint32_t aligned_width = 0, uint32_t aligned_height = 0)
{
    /*
     *
     *  XCAM_ASSERT (native_handle_t.numFds == 1);
     *  XCAM_ASSERT (native_handle_t.data[0] > 0);
     * dma_fd = native_handle_t.data[0] ;
     */;
    VideoBufferInfo info;
    SmartPtr<VideoBuffer> dma_buf;
    SmartPtr<DrmBoBuffer> output;

    XCAM_ASSERT (dma_fd > 0);

    if (aligned_width == 0)
        aligned_width = XCAM_ALIGN_UP(width, 16);
    if (aligned_height == 0)
        aligned_height = XCAM_ALIGN_UP(height, 16);

    info.init (V4L2_PIX_FMT_NV12, width, height, aligned_width, aligned_height, size);
    dma_buf = new DmaVideoBuffer (info, dma_fd);
    output = display->convert_to_drm_bo_buf (display, dma_buf);
    if (!output.ptr ()) {
        XCAM_LOG_ERROR ("dma_buf(%d) convert to xcam_buf failed", dma_fd);
    }

    return output;
}

static SmartPtr<DrmBoBuffer>
create_dma_buffer (SmartPtr<DrmDisplay> &display, const VideoBufferInfo &info)
{
    SmartPtr<BufferPool> buf_pool = new DrmBoBufferPool (display);
    buf_pool->set_video_info (info);
    buf_pool->reserve (1);
    return buf_pool->get_buffer (buf_pool).dynamic_cast_ptr<DrmBoBuffer> ();
}

static XCamReturn
blend_images (
    SmartPtr<DrmBoBuffer> input0, SmartPtr<DrmBoBuffer> input1,
    SmartPtr<DrmBoBuffer> &output_buf,
    SmartPtr<CLBlender> blender)
{
    blender->set_output_size (output_width, output_height);
    input0->attach_buffer (input1);
    return blender->execute (input0, output_buf);
}

int main (int argc, char *argv[])
{
    GeoPos *geo_map0 = NULL, *geo_map1 = NULL;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<CLImageHandler> image_handler;
    SmartPtr<CLGeoMapHandler> geo_map_handler;
    SmartPtr<CLBlender> blender;
    VideoBufferInfo input_buf_info0, input_buf_info1, output_buf_info;
    SmartPtr<CLContext> context;
    SmartPtr<DrmDisplay> display;
    SmartPtr<BufferPool> buf_pool0, buf_pool1;
    ImageFileHandle file_in0, file_in1, file_out;
    SmartPtr<DrmBoBuffer> input0, input1;
    SmartPtr<DrmBoBuffer> output_buf;
    SmartPtr<VideoBuffer> read_buf;

#define FAILED_GEO_FREE  { delete [] geo_map0; delete [] geo_map1; return -1; }

    const struct option long_opts[] = {
        {"input0", required_argument, NULL, 'i'},
        {"input1", required_argument, NULL, 'I'},
        {"output", required_argument, NULL, 'o'},
        {"input-w0", required_argument, NULL, 'w'},
        {"input-w1", required_argument, NULL, 'W'},
        {"input-h", required_argument, NULL, 'H'},
        {"output-w", required_argument, NULL, 'x'},
        {"loop", required_argument, NULL, 'l'},
        {"save", required_argument, NULL, 's'},
        {"enable-geo", no_argument, NULL, 'g'},
        {"enable-seam", no_argument, NULL, 'm'},
        {"help", no_argument, NULL, 'h'},
        {0, 0, 0, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            strncpy (file_in0_name, optarg, XCAM_MAX_STR_SIZE);
            break;
        case 'I':
            strncpy (file_in1_name, optarg, XCAM_MAX_STR_SIZE);
            break;
        case 'o':
            strncpy (file_out_name, optarg, XCAM_MAX_STR_SIZE);
            break;
        case 'w':
            input_width0 = atoi(optarg);
            break;
        case 'W':
            input_width1 = atoi(optarg);
            break;
        case 'H':
            input_height = atoi(optarg);
            break;
        case 'x':
            output_width = atoi(optarg);
            break;
        case 'l':
            loop = atoi(optarg);
            break;
        case 's':
            need_save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'g':
            enable_geo = true;
            break;
        case 'm':
            enable_seam = true;
            break;
        case 'h':
            usage (argv[0]);
            return -1;
        default:
            printf ("getopt_long return unknown value:%c\n", opt);
            usage (argv[0]);
            return -1;

        }
    }

    if (optind < argc || argc < 2) {
        printf("unknown option %s\n", argv[optind]);
        usage (argv[0]);
        return -1;
    }

    printf ("Description-----------\n");
    printf ("input0 file:%s\n", file_in0_name);
    printf ("input1 file:%s\n", file_in1_name);
    printf ("output file PREFIX:%s\n", file_out_name);
    printf ("input0 width:%d\n", input_width0);
    printf ("input1 width:%d\n", input_width1);
    printf ("input/output height:%d\n", input_height);
    printf ("output width:%d\n", output_width);
    printf ("loop count:%d\n", loop);
    printf ("need save file:%s\n", need_save_output ? "true" : "false");
    printf ("enable seam mask:%s\n", (enable_seam ? "true" : "false"));
    printf ("----------------------\n");

    output_height = input_height;
    input_buf_info0.init (input_format, input_width0, input_height);
    input_buf_info1.init (input_format, input_width1, input_height);
    output_buf_info.init (input_format, output_width, output_height);
    display = DrmDisplay::instance ();
    buf_pool0 = new DrmBoBufferPool (display);
    buf_pool1 = new DrmBoBufferPool (display);
    XCAM_ASSERT (buf_pool0.ptr () && buf_pool1.ptr ());
    buf_pool0->set_video_info (input_buf_info0);
    buf_pool1->set_video_info (input_buf_info1);
    if (!buf_pool0->reserve (2)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }
    if (!buf_pool1->reserve (2)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    context = CLDevice::instance ()->get_context ();
    blender = create_pyramid_blender (context, 2, true, enable_seam).dynamic_cast_ptr<CLBlender> ();
    XCAM_ASSERT (blender.ptr ());

#if ENABLE_DMA_TEST
    int dma_fd0 = 30, dma_fd1 = 31, dma_fd_out = 32;
    input_buf_info0.init (
        input_format, input_width0, input_height, XCAM_ALIGN_UP (input_width0, 16), XCAM_ALIGN_UP(input_height, 16));
    input_buf_info1.init (
        input_format, input_width1, input_height, XCAM_ALIGN_UP (input_width1, 16), XCAM_ALIGN_UP(input_height, 16));
    output_buf_info.init (
        input_format, output_width, output_height, XCAM_ALIGN_UP (output_width, 16), XCAM_ALIGN_UP(output_height, 16));
    uint32_t in_size = input_buf_info0.aligned_width * input_buf_info0.aligned_height * 3 / 2;
    uint32_t out_size = output_buf_info.aligned_width * output_buf_info.aligned_height * 3 / 2;
    /* create dma fd, for buffer_handle_t just skip this segment, directly goto dma_buf_to_xcam_buf */
    SmartPtr<DrmBoBuffer> dma_buf0, dma_buf1, dma_buf_out;
    dma_buf0 = create_dma_buffer (display, input_buf_info0); //unit test
    dma_buf1 = create_dma_buffer (display, input_buf_info1); //unit test
    dma_buf_out = create_dma_buffer (display, output_buf_info); //unit test
    dma_fd0 = dma_buf0->get_fd (); //unit test
    dma_fd1 = dma_buf1->get_fd (); //unit test
    dma_fd_out = dma_buf_out->get_fd (); //unit test
    /*
      buffer_handle_t just go to here,
      dma_fd0 = native_handle_t.data[0];
      dma_fd1 = native_handle_t.data[0];
      dma_fd_out = native_handle_t.data[0];
     */
    printf ("DMA handles, buf0:%d, buf1:%d, buf_out:%d\n", dma_fd0, dma_fd1, dma_fd_out);
    input0 = dma_buf_to_xcam_buf (
                 display, dma_fd0, input_width0, input_height, in_size,
                 input_buf_info0.aligned_width, input_buf_info0.aligned_height);
    input1 = dma_buf_to_xcam_buf (
                 display, dma_fd1, input_width0, input_height, in_size,
                 input_buf_info1.aligned_width, input_buf_info1.aligned_height);
    output_buf = dma_buf_to_xcam_buf (
                     display, dma_fd_out, output_width, output_height, out_size,
                     output_buf_info.aligned_width, output_buf_info.aligned_height);
    blender->disable_buf_pool (true);
#else
    input0 = buf_pool0->get_buffer (buf_pool0).dynamic_cast_ptr<DrmBoBuffer> ();
    input1 = buf_pool1->get_buffer (buf_pool1).dynamic_cast_ptr<DrmBoBuffer> ();
    XCAM_ASSERT (input0.ptr () && input1.ptr ());
#endif
    //
    ret = file_in0.open (file_in0_name, "rb");
    CHECK_STATEMENT (ret, FAILED_GEO_FREE, "open input file(%s) failed", file_in0_name);
    read_buf = input0;
    ret = file_in0.read_buf (read_buf);
    CHECK_STATEMENT (ret, FAILED_GEO_FREE, "read buffer0 from (%s) failed", file_in0_name);

    ret = file_in1.open (file_in1_name, "rb");
    CHECK_STATEMENT (ret, FAILED_GEO_FREE, "open input file(%s) failed", file_in1_name);
    read_buf = input1;
    ret = file_in1.read_buf (read_buf);
    CHECK_STATEMENT (ret, FAILED_GEO_FREE, "read buffer1 from (%s) failed", file_in1_name);

    if (enable_geo) {
        geo_map_handler = create_geo_map_handler (context).dynamic_cast_ptr<CLGeoMapHandler> ();
        XCAM_ASSERT (geo_map_handler.ptr ());

        geo_map0 = new GeoPos[map_width * map_height];
        geo_map1 = new GeoPos[map_width * map_height];
        XCAM_ASSERT (geo_map0 && geo_map1);
        if (read_map_data (map0, geo_map0, map_width, map_height) <= 0 ||
                read_map_data (map1, geo_map1, map_width, map_height) <= 0) {
            delete [] geo_map0;
            delete [] geo_map1;
            return -1;
        }

        geo_map_handler->set_map_uint (28.0f, 28.0f);
    }

    int i = 0;
    do {
        input0->clear_attached_buffers ();
        input1->clear_attached_buffers ();

        if (enable_geo) {
            geo_correct_image (geo_map_handler, input0, geo_map0, map_width, map_height, file_in0_name, need_save_output);
            geo_correct_image (geo_map_handler, input1, geo_map1, map_width, map_height, file_in1_name, need_save_output);
        }

        ret = blend_images (input0, input1, output_buf, blender);
        CHECK_STATEMENT (ret, FAILED_GEO_FREE, "blend_images execute failed");
        //printf ("DMA handles, output_buf:%d\n", output_buf->get_fd ());

        if (need_save_output) {
            char out_name[1024];
            snprintf (out_name, 1023, "%s.%02d", file_out_name, i);

            ret = file_out.open (out_name, "wb");
            CHECK_STATEMENT (ret, FAILED_GEO_FREE, "open output file(%s) failed", out_name);
            ret = file_out.write_buf (output_buf);
            CHECK_STATEMENT (ret, FAILED_GEO_FREE, "write buffer to (%s) failed", out_name);
            printf ("write output buffer to: %s done\n", out_name);
        } else {
            // check info
            ensure_gpu_buffer_done (output_buf);
        }

        FPS_CALCULATION (image_blend, XCAM_OBJ_DUR_FRAME_NUM);
        ++i;
    } while (i < loop);

    delete [] geo_map0;
    delete [] geo_map1;

    return ret;
}

//return count
int read_map_data (const char* file, GeoPos *map, int width, int height)
{
    char *ptr = NULL;
    FILE *p_f = fopen (file, "rb");
    CHECK_EXP (p_f, "open geo-map file(%s) failed", file);

#define FAILED_READ_MAP { if (p_f) fclose(p_f); if (ptr) xcam_free (ptr); return -1; }

    CHECK_DECLARE (ERROR, fseek(p_f, 0L, SEEK_END) == 0, FAILED_READ_MAP, "seek to file(%s) end failed", file);
    size_t size = ftell (p_f);
    XCAM_ASSERT ((int)size != -1);
    fseek (p_f, 0L, SEEK_SET);

    ptr = (char*)xcam_malloc (size + 1);
    XCAM_ASSERT (ptr);
    CHECK_DECLARE (ERROR, fread (ptr, 1, size, p_f) == size, FAILED_READ_MAP, "read map file(%s)failed", file);
    ptr[size] = 0;
    fclose (p_f);
    p_f = NULL;

    char *str_num = NULL;
    char tokens[] = "\t ,\r\n";
    str_num = strtok (ptr, tokens);
    int count = 0;
    int x = 0, y = 0;
    while (str_num != NULL) {
        float num = strtof (str_num, NULL);
        //printf ("%.3f\n", num);

        x = count % width;
        y = count / (width * 2); // x,y
        if (y >= height)
            break;

        if (count % (width * 2) >= width)
            map[y * width + x].y = num;
        else
            map[y * width + x].x = num;

        ++count;
        str_num = strtok (NULL, tokens);
    }
    xcam_free (ptr);
    ptr = NULL;
    CHECK_EXP (y < height, "map data(%s) count larger than expected(%dx%dx2)", file, width, height);
    CHECK_EXP (count >= width * height * 2, "map data(%s) count less than expected(%dx%dx2)", file, width, height);

    printf ("read map(%s) x/y data count:%d\n", file, count);
    return count;
}

