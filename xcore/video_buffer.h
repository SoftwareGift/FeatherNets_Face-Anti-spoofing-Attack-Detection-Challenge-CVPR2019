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

namespace XCam {

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

#define XCAM_VIDEO_MAX_COMPONENTS 4

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
        uint32_t alignment_w = 4, uint32_t alignment_h = 2);
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

private:
    VideoBufferInfo _videoinfo;
    int64_t         _timestamp; // in microseconds
};

};

#endif //XCAM_BUFFER_PROXY_H
