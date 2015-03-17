/*
 * stub.h - stub utilities that implemented in CPP
 *
 *  Copyright (c) 2015 Intel Corporation
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
 * Author: John Ye <john.ye@intel.com>
 */

#ifndef __STUB_H__
#define __STUB_H__

#include <unistd.h>
#include <stdint.h>
#include <linux/videodev2.h>

#include <gst/gst.h>
#include <gst/allocators/allocators.h>
#include <gst/video/gstvideopool.h>
#include "gstxcambufferpool.h"

#ifdef __cplusplus
extern "C" {
#endif

enum v4l2_memory;
enum v4l2_field;
struct v4l2_format;
struct v4l2_buffer;

int libxcam_set_device_name (const char* ch);
int libxcam_set_sensor_id (int id);
int libxcam_set_capture_mode (uint32_t cap_mode);
int libxcam_set_mem_type (enum v4l2_memory mem_type);
int libxcam_set_buffer_count (uint32_t buf_count);
int libxcam_set_framerate (uint32_t fps_n, uint32_t fps_d);
int libxcam_open ();
int libxcam_close ();
int libxcam_set_format (uint32_t width, uint32_t height, uint32_t pixelformat, enum v4l2_field field, uint32_t bytes_perline);
int libxcam_get_blocksize (uint32_t *blocksize);
int libxcam_start ();
int libxcam_stop ();
GstFlowReturn xcam_bufferpool_acquire_buffer (GstBufferPool *bpool, GstBuffer **buffer, GstBufferPoolAcquireParams *params);
void xcambufferpool_release_buffer (GstBufferPool *bpool, GstBuffer *buffer);

#ifdef __cplusplus
}
#endif

#endif  //__STUB_H__
