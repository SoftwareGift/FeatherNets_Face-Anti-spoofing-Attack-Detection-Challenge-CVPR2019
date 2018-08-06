/*
 * test_vk_handler.cpp - test vulkan handler
 *
 *  Copyright (c) 2018 Intel Corporation
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
#include "image_file_handle.h"
#include "buffer_pool.h"
#include "vulkan/vk_device.h"
#include "vulkan/vk_copy_handler.h"

using namespace XCam;

static void
print_help (const char *bin_name)
{
    printf ("Usage: %s -i input -o output\n"
            "\t -W image_width    specify input image width, default 1920\n"
            "\t -H image_height   specify input image height, default 1080\n"
            "\t -f input_format   specify a input format\n"
            "\t                   select from [NV12, BA10, RGBA, RGBA64], default NV12\n"
            "\t -i input          specify input file path\n"
            "\t -o output         specify output file path\n"
            "\t -h                help\n"
            , bin_name);

    printf ("Note:\n"
            "Spirv path Setup Env: $" XCAM_VK_SHADER_PATH "\n"
            "Generate spirv kernel:\n"
            "glslangValidator -V -x -o sample.comp.spv sample.comp.sl\n"
           );
}

int main (int argc, char *argv[])
{
    uint32_t input_format = V4L2_PIX_FMT_NV12;
    //uint32_t output_format = input_format;
    uint32_t width = 1920;
    uint32_t height = 1080;
    std::string input_file, output_file;
    ImageFileHandle input_fp, output_fp;
    const char *bin_name = argv[0];
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    SmartPtr<ImageHandler> image_handler;
    VideoBufferInfo input_buf_info;

    SmartPtr<VKDevice> vk_device = VKDevice::default_device ();
    XCAM_FAIL_RETURN (
        ERROR, vk_device.ptr(), -1,
        "Get default VKDevice failed, please check vulkan environment.");

    int opt = 0;
    while ((opt =  getopt(argc, argv, "f:W:H:i:o:Ph")) != -1) {
        switch (opt) {
        case 'i':
            input_file = optarg;
            break;
        case 'o':
            output_file = optarg;
            break;

        case 'f': {
            if (!strcasecmp (optarg, "nv12"))
                input_format = V4L2_PIX_FMT_NV12;
            else if (! strcasecmp (optarg, "rgba"))
                input_format = V4L2_PIX_FMT_RGBA32;
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
        case 'h':
            print_help (bin_name);
            return 0;

        default:
            print_help (bin_name);
            return -1;
        }
    }

    if (!input_format || !input_file.length() || !output_file.length() ) {
        print_help (bin_name);
        return -1;
    }

    ret = input_fp.open (input_file.c_str(), "rb");
    CHECK (ret, "open input file(%s) failed", input_file.c_str());
    ret = output_fp.open (output_file.c_str(), "wb");
    CHECK (ret, "open output file(%s) failed", output_file.c_str());

    SmartPtr<BufferPool> buf_pool = create_vk_buffer_pool (vk_device);
    CHECK_EXP (buf_pool.ptr (), "vk buffer pool create failed");
    VideoBufferInfo in_buf_info;
    in_buf_info.init (input_format, width, height);
    buf_pool->set_video_info (in_buf_info);
    buf_pool->reserve (2);
    SmartPtr<VideoBuffer> in_buf = buf_pool->get_buffer ();
    SmartPtr<VideoBuffer> out_buf = buf_pool->get_buffer ();
    CHECK_EXP (in_buf.ptr () && out_buf.ptr (), "vk buffer (in/out) was not allocated");
    CHECK (input_fp.read_buf (in_buf), "read buf from file:%s failed", input_file.c_str());

    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters(in_buf, out_buf);
    image_handler = new VKCopyHandler (vk_device, "test-vk-copy");
    XCAM_ASSERT (image_handler.ptr ());
    CHECK (
        image_handler->execute_buffer (param, true),
        "handler:(%s) execute buffer failed.", XCAM_STR (image_handler->get_name ()));

    XCAM_ASSERT (out_buf.ptr ());
    XCAM_LOG_INFO ("writing out buf to file:%s", output_file.c_str());
    CHECK (output_fp.write_buf (out_buf), "write buf to file: %s failed", output_file.c_str());
#if 1
    uint8_t * in_data = in_buf->map ();
    uint8_t * out_data = out_buf->map ();
    XCAM_ASSERT (in_data && out_data);
    for (size_t i = 0; i < in_buf_info.size; ++i) {
        CHECK_EXP (in_data[i] == out_data[i], "vk copy buffer error, in and out buf data not match:(pos:%d)", i);
    }
    XCAM_LOG_INFO ("test passed.");
#endif
    in_buf->unmap ();
    out_buf->unmap ();

    return 0;
}

