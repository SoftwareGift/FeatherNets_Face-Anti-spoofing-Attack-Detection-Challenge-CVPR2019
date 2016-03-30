/*
 * xcam_buffer.h - video buffer standard version
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

#ifndef C_XCAM_BUFFER_H
#define C_XCAM_BUFFER_H

#include <base/xcam_common.h>

XCAM_BEGIN_DECLARE
#include <linux/videodev2.h>

#ifndef V4L2_PIX_FMT_XBGR32
#define V4L2_PIX_FMT_XBGR32 v4l2_fourcc('X', 'R', '2', '4')
#endif

#ifndef V4L2_PIX_FMT_ABGR32
#define V4L2_PIX_FMT_ABGR32 v4l2_fourcc('A', 'R', '2', '4')
#endif

#ifndef V4L2_PIX_FMT_XRGB32
#define V4L2_PIX_FMT_XRGB32 v4l2_fourcc('B', 'X', '2', '4')
#endif

#ifndef V4L2_PIX_FMT_ARGB32
#define V4L2_PIX_FMT_ARGB32 v4l2_fourcc('B', 'A', '2', '4')
#endif

#ifndef V4L2_PIX_FMT_RGBA32
#define V4L2_PIX_FMT_RGBA32 v4l2_fourcc('A', 'B', '2', '4')
#endif

/*
 * Define special format for 16 bit color
 * every format start with 'X'
 *
 * XCAM_PIX_FMT_RGB48: RGB with color-bits = 16
 * XCAM_PIX_FMT_RGBA64, RGBA with color-bits = 16
 * XCAM_PIX_FMT_SGRBG16, Bayer, with color-bits = 16
 */

#define XCAM_PIX_FMT_RGB48     v4l2_fourcc('w', 'R', 'G', 'B')
#define XCAM_PIX_FMT_RGBA64     v4l2_fourcc('w', 'R', 'G', 'a')
#define XCAM_PIX_FMT_SGRBG16   v4l2_fourcc('w', 'B', 'A', '0')
#define XCAM_PIX_FMT_LAB    v4l2_fourcc('h', 'L', 'a', 'b')
#define XCAM_PIX_FMT_RGB48_planar     v4l2_fourcc('n', 'R', 'G', 0x48)
#define XCAM_PIX_FMT_RGB24_planar     v4l2_fourcc('n', 'R', 'G', 0x24)
#define XCAM_PIX_FMT_SGRBG16_planar   v4l2_fourcc('n', 'B', 'A', '0')
#define XCAM_PIX_FMT_SGRBG8_planar   v4l2_fourcc('n', 'B', 'A', '8')

#define XCAM_VIDEO_MAX_COMPONENTS 4


typedef struct _XCamVideoBufferPlanarInfo XCamVideoBufferPlanarInfo;
struct _XCamVideoBufferPlanarInfo {
    uint32_t width;
    uint32_t height;
    uint32_t pixel_bytes;
};

typedef struct _XCamVideoBufferInfo XCamVideoBufferInfo;
struct _XCamVideoBufferInfo {
    uint32_t format; // v4l2 fourcc
    uint32_t color_bits;
    uint32_t width;
    uint32_t height;
    uint32_t aligned_width;
    uint32_t aligned_height;
    uint32_t size;
    uint32_t components;
    uint32_t strides [XCAM_VIDEO_MAX_COMPONENTS];
    uint32_t offsets [XCAM_VIDEO_MAX_COMPONENTS];
};

typedef enum {
    XCAM_MEM_TYPE_CPU,
    XCAM_MEM_TYPE_GPU,
    XCAM_MEM_TYPE_PRIVATE = 0x8000,
    XCAM_MEM_TYPE_PRIVATE_BO,
} XCamMemType;

typedef struct _XCamVideoBuffer XCamVideoBuffer;

struct _XCamVideoBuffer {
    XCamVideoBufferInfo   info;
    uint32_t              mem_type;
    int64_t               timestamp;

    void      (*ref) (XCamVideoBuffer *);
    void      (*unref) (XCamVideoBuffer *);
    uint8_t  *(*map) (XCamVideoBuffer *);
    void      (*unmap) (XCamVideoBuffer *);
    int       (*get_fd) (XCamVideoBuffer *);
};

typedef struct _XCamVideoBufferIntel XCamVideoBufferIntel;
struct _XCamVideoBufferIntel {
    XCamVideoBuffer     base;

    void     *(*get_bo) (XCamVideoBufferIntel *);
};

#define xcam_video_buffer_ref(buf) (buf)->ref(buf)
#define xcam_video_buffer_unref(buf) (buf)->unref(buf)
#define xcam_video_buffer_map(buf) (buf)->map(buf)
#define xcam_video_buffer_unmap(buf) (buf)->unmap(buf)
#define xcam_video_buffer_get_fd(buf) (buf)->get_fd(buf)
#define xcam_video_buffer_intel_get_bo(buf) (buf)->get_bo(buf)

XCamReturn
xcam_video_buffer_info_reset (
    XCamVideoBufferInfo *info,
    uint32_t format, uint32_t width, uint32_t height,
    uint32_t aligned_width, uint32_t aligned_height, uint32_t size);

XCamReturn
xcam_video_buffer_get_planar_info (
    const XCamVideoBufferInfo *buf_info,  XCamVideoBufferPlanarInfo *planar_info, const uint32_t index);


XCAM_END_DECLARE

#endif // C_XCAM_BUFFER_H
