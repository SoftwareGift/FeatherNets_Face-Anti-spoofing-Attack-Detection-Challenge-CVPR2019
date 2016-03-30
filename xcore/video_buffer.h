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
#include "base/xcam_buffer.h"
#include <list>

namespace XCam {

class VideoBuffer;
typedef std::list<SmartPtr<VideoBuffer>>  VideoBufferList;

struct VideoBufferPlanarInfo
        : XCamVideoBufferPlanarInfo
{
    VideoBufferPlanarInfo ();
};

struct VideoBufferInfo
        : XCamVideoBufferInfo
{
    VideoBufferInfo ();
    bool init (
        uint32_t format,
        uint32_t width, uint32_t height,
        uint32_t aligned_width = 0, uint32_t aligned_height = 0, uint32_t size = 0);

    bool get_planar_info (
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
