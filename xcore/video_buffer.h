/*
 * video_buffer.h - video buffer base
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

#ifndef XCAM_VIDEO_BUFFER_H
#define XCAM_VIDEO_BUFFER_H

#include "xcam_utils.h"
#include "smartptr.h"
#include <list>

namespace XCam {

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

#define XCAM_VIDEO_MAX_COMPONENTS 4

class VideoBuffer;
typedef std::list<SmartPtr<VideoBuffer>>  VideoBufferList;

struct VideoBufferPlanarInfo {
    uint32_t width;
    uint32_t height;
    uint32_t pixel_bytes;

    VideoBufferPlanarInfo ();
};

struct VideoBufferInfo {
    uint32_t format;
    uint32_t color_bits;
    uint32_t width;
    uint32_t height;
    uint32_t aligned_width;
    uint32_t aligned_height;
    uint32_t size;
    uint32_t components;
    uint32_t strides [XCAM_VIDEO_MAX_COMPONENTS];
    uint32_t offsets [XCAM_VIDEO_MAX_COMPONENTS];

    VideoBufferInfo ();
    bool init (
        uint32_t format,
        uint32_t width, uint32_t height,
        uint32_t aligned_width = 0, uint32_t aligned_height = 0, uint32_t size = 0);

    bool get_planar_info (
        const uint32_t format,
        const uint32_t  width, const uint32_t height,
        VideoBufferPlanarInfo &planar, const uint32_t index = 0) const;
};

class VideoBuffer {
public:
    explicit VideoBuffer (int64_t timestamp = InvalidTimestamp)
        : _timestamp (timestamp)
    {}
    explicit VideoBuffer (const VideoBufferInfo &info, int64_t timestamp = InvalidTimestamp)
        : _videoinfo (info)
        , _timestamp (timestamp)
    {}
    virtual ~VideoBuffer () {}

    virtual uint8_t *map () = 0;
    virtual bool unmap () = 0;
    virtual int get_fd () = 0;

    const VideoBufferInfo & get_video_info () const {
        return _videoinfo;
    }
    int64_t get_timestamp () const {
        return _timestamp;
    }

    void set_video_info (const VideoBufferInfo &info) {
        _videoinfo = info;
    }

    void set_timestamp (int64_t timestamp) {
        _timestamp = timestamp;
    }

    uint32_t get_size () const {
        return _videoinfo.size;
    }
private:
    VideoBufferInfo _videoinfo;
    int64_t         _timestamp; // in microseconds
};

};

#endif //XCAM_BUFFER_PROXY_H
