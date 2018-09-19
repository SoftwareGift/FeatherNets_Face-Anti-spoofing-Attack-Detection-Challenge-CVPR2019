/*
 * test-video-stabilization.cpp - test video stabilization using Gyroscope
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#include "test_common.h"
#include "test_inline.h"
#include <unistd.h>
#include <getopt.h>
#include <ocl/cl_utils.h>
#include <ocl/cl_device.h>
#include <ocl/cl_context.h>
#include <ocl/cl_blender.h>
#include <image_file_handle.h>
#include <ocl/cl_video_stabilizer.h>
#include <dma_video_buffer.h>

#if HAVE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <ocl/cv_base_class.h>
#endif

using namespace XCam;

static int read_device_pose (const char *file, DevicePoseList &pose, uint32_t pose_size);

static void
usage(const char* arg0)
{
    printf ("Usage:\n"
            "%s --input file --output file"
            " [--input-w width] [--input-h height] \n"
            "\t--input, input image(NV12)\n"
            "\t--gyro, input gyro pose data;\n"
            "\t--output, output image(NV12) PREFIX\n"
            "\t--input-w, optional, input width; default:1920\n"
            "\t--input-h,  optional, input height; default:1080\n"
            "\t--save,     optional, save file or not, default true; select from [true/false]\n"
            "\t--loop      optional, how many loops need to run for performance test, default: 1\n"
            "\t--help,     usage\n",
            arg0);
}

int main (int argc, char *argv[])
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    SmartPtr<CLVideoStabilizer> video_stab;
    SmartPtr<CLContext> context;
    VideoBufferInfo input_buf_info;
    VideoBufferInfo output_buf_info;
    SmartPtr<VideoBuffer> input_buf;
    SmartPtr<VideoBuffer> output_buf;

    uint32_t input_format = V4L2_PIX_FMT_NV12;
    uint32_t input_width = 1920;
    uint32_t input_height = 1080;
    uint32_t output_width = 1920;
    uint32_t output_height = 1080;

    ImageFileHandle file_in, file_out;
    const char *file_in_name = NULL;
    const char *file_out_name = NULL;

    const char *gyro_data = "gyro_data.csv";

    bool need_save_output = true;
    double framerate = 30.0;
    int loop = 1;

    const struct option long_opts[] = {
        {"input", required_argument, NULL, 'i'},
        {"gyro", required_argument, NULL, 'g'},
        {"output", required_argument, NULL, 'o'},
        {"input-w", required_argument, NULL, 'w'},
        {"input-h", required_argument, NULL, 'h'},
        {"save", required_argument, NULL, 's'},
        {"loop", required_argument, NULL, 'l'},
        {"help", no_argument, NULL, 'H'},
        {0, 0, 0, 0},
    };

    int opt = -1;
    while ((opt = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i':
            file_in_name = optarg;
            break;
        case 'g':
            gyro_data = optarg;
            break;
        case 'o':
            file_out_name = optarg;
            break;
        case 'w':
            input_width = atoi(optarg);
            output_width = input_width;
            break;
        case 'h':
            input_height = atoi(optarg);
            output_height = input_height;
            break;
        case 's':
            need_save_output = (strcasecmp (optarg, "false") == 0 ? false : true);
            break;
        case 'l':
            loop = atoi(optarg);
            break;
        case 'H':
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

    if (!file_in_name || !file_out_name) {
        XCAM_LOG_ERROR ("input/output path is NULL");
        return -1;
    }

    printf ("Description-----------\n");
    printf ("input video file:%s\n", file_in_name);
    printf ("gyro pose file:%s\n", gyro_data);
    printf ("output file PREFIX:%s\n", file_out_name);
    printf ("input width:%d\n", input_width);
    printf ("input height:%d\n", input_height);
    printf ("need save file:%s\n", need_save_output ? "true" : "false");
    printf ("loop count:\t\t%d\n", loop);
    printf ("----------------------\n");

    DevicePoseList device_pose;

    const int pose_size = sizeof(DevicePose::orientation) / sizeof(double) +
                          sizeof(DevicePose::translation) / sizeof(double) +
                          sizeof(DevicePose::timestamp) / sizeof(int64_t);

    const int count = read_device_pose (gyro_data, device_pose, pose_size);
    if (count <= 0 || device_pose.size () <= 0) {
        XCAM_LOG_ERROR ("read gyro file(%s) failed.", gyro_data);
        return -1;
    }

    context = CLDevice::instance ()->get_context ();
    video_stab = create_cl_video_stab_handler (context).dynamic_cast_ptr<CLVideoStabilizer> ();
    XCAM_ASSERT (video_stab.ptr ());
    video_stab->set_pool_type (CLImageHandler::CLVideoPoolType);

    /*
        Color CameraIntrinsics:
                 image_width: 1920, image_height :1080,
                 fx: 1707.799171, fy: 1710.337510,
                 cx: 940.413257, cy: 540.198348,
                 image_plane_distance: 1.778957.

        Color Camera Frame with respect to IMU Frame:
                 Position: 0.045699, -0.008592, -0.006434
                 Orientation: -0.013859, -0.999889, 0.002361, 0.005021
    */
    double focal_x = 1707.799171;
    double focal_y = 1710.337510;
    double offset_x = 940.413257;
    double offset_y = 540.198348;
    double skew = 0;
    video_stab->set_camera_intrinsics (focal_x, focal_y, offset_x, offset_y, skew);

    CoordinateSystemConv world_to_device (AXIS_X, AXIS_MINUS_Z, AXIS_NONE);
    CoordinateSystemConv device_to_image (AXIS_X, AXIS_Y, AXIS_Y);
    video_stab->align_coordinate_system (world_to_device, device_to_image);

    uint32_t radius = 15;
    float stdev = 10;
    video_stab->set_motion_filter (radius, stdev);

    input_buf_info.init (input_format, input_width, input_height);
    output_buf_info.init (input_format, output_width, output_height);
    SmartPtr<BufferPool> buf_pool = new CLVideoBufferPool ();
    XCAM_ASSERT (buf_pool.ptr ());
    buf_pool->set_video_info (input_buf_info);
    if (!buf_pool->reserve (36)) {
        XCAM_LOG_ERROR ("init buffer pool failed");
        return -1;
    }

    ret = file_in.open (file_in_name, "rb");
    CHECK (ret, "open %s failed", file_in_name);

#if HAVE_OPENCV
    cv::VideoWriter writer;
    if (need_save_output) {
        cv::Size dst_size = cv::Size (output_width, output_height);
        if (!writer.open (file_out_name, cv::VideoWriter::fourcc ('X', '2', '6', '4'), framerate, dst_size)) {
            XCAM_LOG_ERROR ("open file %s failed", file_out_name);
            return -1;
        }
    }
#endif

    int i = 0;
    while (loop--) {
        ret = file_in.rewind ();
        CHECK (ret, "video stabilization stitch rewind file(%s) failed", file_in_name);

        video_stab->reset_counter ();

        DevicePoseList::iterator pose_iterator = device_pose.begin ();
        do {
            input_buf = buf_pool->get_buffer (buf_pool);
            XCAM_ASSERT (input_buf.ptr ());
            ret = file_in.read_buf (input_buf);
            if (ret == XCAM_RETURN_BYPASS)
                break;
            if (ret == XCAM_RETURN_ERROR_FILE) {
                XCAM_LOG_ERROR ("read buffer from %s failed", file_in_name);
                return -1;
            }

            SmartPtr<MetaData> pose_data  = *(pose_iterator);
            SmartPtr<DevicePose> data = *(pose_iterator);
            input_buf->add_metadata (pose_data);
            input_buf->set_timestamp (pose_data->timestamp);

            ret = video_stab->execute (input_buf, output_buf);
            if (++pose_iterator == device_pose.end ()) {
                break;
            }
            if (ret == XCAM_RETURN_BYPASS) {
                continue;
            }

#if HAVE_OPENCV
            if (need_save_output) {
                cv::Mat out_mat;
                convert_to_mat (output_buf, out_mat);
                writer.write (out_mat);
            } else
#endif
                ensure_gpu_buffer_done (output_buf);

            FPS_CALCULATION (video_stabilizer, XCAM_OBJ_DUR_FRAME_NUM);
            ++i;

        } while (true);
    }

    return ret;
}

//return count

#define RELEASE_FILE_MEM {      \
        xcam_free (ptr);        \
        if (p_f) fclose (p_f);  \
        return -1;              \
    }

int read_device_pose (const char* file, DevicePoseList &pose_list, uint32_t members)
{
    char *ptr = NULL;
    SmartPtr<DevicePose> data;

    FILE *p_f = fopen (file, "rb");
    CHECK_EXP (p_f, "open gyro pos data(%s) failed", file);

    CHECK_DECLARE (
        ERROR,
        !fseek (p_f, 0L, SEEK_END),
        RELEASE_FILE_MEM, "seek to file(%s) end failed", file);

    size_t size = ftell(p_f);
    int entries = size / members;

    fseek (p_f, 0L, SEEK_SET);

    ptr = (char*) xcam_malloc0 (size + 1);
    CHECK_DECLARE (ERROR, ptr, RELEASE_FILE_MEM, "malloc file buffer failed");

    CHECK_DECLARE (
        ERROR,
        fread (ptr, 1, size, p_f) == size,
        RELEASE_FILE_MEM, "read pose file(%s)failed", file);
    ptr[size] = 0;
    fclose (p_f);
    p_f = NULL;

    char *str_num = NULL;
    char tokens[] = "\t ,\r\n";
    str_num = strtok (ptr, tokens);
    int count = 0;
    int x = 0, y = 0;
    const int orient_size = sizeof(DevicePose::orientation) / sizeof(double);
    const int trans_size = sizeof(DevicePose::translation) / sizeof(double);

    while (str_num != NULL) {
        float num = strtof (str_num, NULL);

        x = count % members;
        y = count / members;
        if (y >= entries) {
            break;
        }
        if (x == 0) {
            data = new DevicePose ();
        }

        CHECK_DECLARE (ERROR, data.ptr (), RELEASE_FILE_MEM, "invalid buffer pointer(device pose is null)");
        if (x < orient_size) {
            data->orientation[x] = num;
        } else if (x < orient_size + trans_size) {
            data->translation[x - orient_size] = num;
        } else if (x == orient_size + trans_size) {
            data->timestamp = num * 1000000;
            pose_list.push_back (data);
        } else {
            CHECK_DECLARE (ERROR, false, RELEASE_FILE_MEM, "unknow branch");
        }

        ++count;
        str_num = strtok (NULL, tokens);
    }
    free (ptr);
    ptr = NULL;

    return count / members;
}

