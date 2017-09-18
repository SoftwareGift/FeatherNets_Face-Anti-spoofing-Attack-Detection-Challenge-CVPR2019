/*
 * soft_video_buf_allocator.cpp - soft video buffer allocator implementation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "soft_video_buf_allocator.h"

namespace XCam {

class VideoMemData
    : public BufferData
{
public:
    explicit VideoMemData (uint32_t size);
    virtual ~VideoMemData ();
    bool is_valid () const {
        return (_mem_ptr ? true : false);
    }

    //derive from BufferData
    virtual uint8_t *map ();
    virtual bool unmap ();

private:
    uint8_t    *_mem_ptr;
    uint32_t    _mem_size;
};

VideoMemData::VideoMemData (uint32_t size)
    : _mem_ptr (NULL)
    , _mem_size (0)
{
    XCAM_ASSERT (size > 0);
    _mem_ptr = xcam_malloc_type_array (uint8_t, size);
    if (_mem_ptr)
        _mem_size = size;
}

VideoMemData::~VideoMemData ()
{
    xcam_free (_mem_ptr);
}

uint8_t *
VideoMemData::map ()
{
    XCAM_ASSERT (_mem_ptr);
    return _mem_ptr;
}

bool
VideoMemData::unmap ()
{
    return true;
}

SoftVideoBufAllocator::SoftVideoBufAllocator ()
{
}

SoftVideoBufAllocator::SoftVideoBufAllocator (const VideoBufferInfo &info)
{
    set_video_info (info);
}

SoftVideoBufAllocator::~SoftVideoBufAllocator ()
{
}

SmartPtr<BufferData>
SoftVideoBufAllocator::allocate_data (const VideoBufferInfo &buffer_info)
{
    XCAM_FAIL_RETURN (
        ERROR, buffer_info.size, NULL,
        "SoftVideoBufAllocator allocate data failed. buf_size is zero");

    SmartPtr<VideoMemData> data = new VideoMemData (buffer_info.size);
    XCAM_FAIL_RETURN (
        ERROR, data.ptr () && data->is_valid (), NULL,
        "SoftVideoBufAllocator allocate data failed. buf_size:%d", buffer_info.size);

    return data;
}

}

