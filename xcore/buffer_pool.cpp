/*
 * buffer_pool.cpp - buffer pool
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

#include "xcam_utils.h"
#include "buffer_pool.h"

namespace XCam {

BufferProxy::BufferProxy (const VideoBufferInfo &info, const SmartPtr<BufferData> &data)
    : VideoBuffer (info)
    , _data (data)
{
    XCAM_ASSERT (data.ptr ());
}

BufferProxy::BufferProxy (const SmartPtr<BufferData> &data)
    : _data (data)
{
    XCAM_ASSERT (data.ptr ());
}

BufferProxy::~BufferProxy ()
{
    clear_attached_buffers ();

    if (_pool.ptr ()) {
        _pool->release (_data);
    }
    _data.release ();
    _parent.release ();
}

uint8_t *
BufferProxy::map ()
{
    XCAM_ASSERT (_data.ptr ());
    return _data->map ();
}

bool
BufferProxy::unmap ()
{
    XCAM_ASSERT (_data.ptr ());
    return _data->unmap ();
}

int
BufferProxy::get_fd ()
{
    XCAM_ASSERT (_data.ptr ());
    return _data->get_fd ();
}

bool
BufferProxy::attach_buffer (const SmartPtr<VideoBuffer>& buf)
{
    _attached_bufs.push_back (buf);
    return true;
}

bool
BufferProxy::detach_buffer (const SmartPtr<VideoBuffer>& buf)
{
    for (VideoBufferList::iterator iter = _attached_bufs.begin ();
            iter != _attached_bufs.end (); ++iter) {
        SmartPtr<VideoBuffer>& current = *iter;
        if (current.ptr () == buf.ptr ()) {
            _attached_bufs.erase (iter);
            return true;
        }
    }

    //not found
    return false;
}

bool
BufferProxy::copy_attaches (const SmartPtr<BufferProxy>& buf)
{
    _attached_bufs.insert (
        _attached_bufs.end (), buf->_attached_bufs.begin (), buf->_attached_bufs.end ());
    return true;
}

void
BufferProxy::clear_attached_buffers ()
{
    _attached_bufs.clear ();
}

bool
BufferProxy::attach_metadata (const SmartPtr<MetaData>& data)
{
    _attached_metadatas.push_back (data);
    return true;
}

bool
BufferProxy::detach_metadata (const SmartPtr<MetaData>& data)
{
    for (MetaDataList::iterator iter = _attached_metadatas.begin ();
            iter != _attached_metadatas.end (); ++iter) {
        SmartPtr<MetaData>& current = *iter;
        if (current.ptr () == data.ptr ()) {
            _attached_metadatas.erase (iter);
            return true;
        }
    }

    //not found
    return false;
}

void
BufferProxy::clear_attached_metadatas ()
{
    _attached_metadatas.clear ();
}

BufferPool::BufferPool ()
    : _allocated_num (0)
    , _max_count (0)
    , _started (false)
{
}

BufferPool::~BufferPool ()
{
}

bool
BufferPool::set_video_info (const VideoBufferInfo &info)
{
    VideoBufferInfo new_info = info;
    SmartLock lock (_mutex);

    XCAM_FAIL_RETURN (
        ERROR,
        fixate_video_info (new_info),
        false,
        "BufferPool fixate video info failed");
    update_video_info_unsafe (new_info);
    return true;
}

void
BufferPool::update_video_info_unsafe (const VideoBufferInfo &info)
{
    _buffer_info = info;
}

bool
BufferPool::reserve (uint32_t max_count)
{
    uint32_t i = 0;

    XCAM_ASSERT (max_count);

    SmartLock lock (_mutex);

    for (i = _allocated_num; i < max_count; ++i) {
        SmartPtr<BufferData> new_data = allocate_data (_buffer_info);
        if (!new_data.ptr ())
            break;
        _buf_list.push (new_data);
    }

    XCAM_FAIL_RETURN (
        ERROR,
        i > 0,
        false,
        "BufferPool reserve failed with none buffer data allocated");

    if (i != max_count) {
        XCAM_LOG_WARNING ("BufferPool expect to reserve %d data but only reserved %d", max_count, i);
    }
    _max_count = i;
    _allocated_num = _max_count;
    _started = true;

    return true;
}

bool
BufferPool::add_data_unsafe (SmartPtr<BufferData> data)
{
    if (!data.ptr ())
        return false;

    _buf_list.push (data);
    ++_allocated_num;

    XCAM_ASSERT (_allocated_num <= _max_count || !_max_count);
    return true;
}

SmartPtr<BufferProxy>
BufferPool::get_buffer (const SmartPtr<BufferPool> &self)
{
    SmartPtr<BufferProxy> ret_buf;
    SmartPtr<BufferData> data;

    {
        SmartLock lock (_mutex);
        if (!_started)
            return NULL;
    }

    XCAM_ASSERT (self.ptr () == this);
    XCAM_FAIL_RETURN(
        WARNING,
        self.ptr () == this,
        NULL,
        "BufferPool get_buffer failed since parameter<self> not this");

    data = _buf_list.pop ();
    if (!data.ptr ()) {
        XCAM_LOG_DEBUG ("BufferPool failed to get buffer");
        return NULL;
    }
    ret_buf = create_buffer_from_data (data);
    ret_buf->set_buf_pool (self);

    return ret_buf;
}

void
BufferPool::stop ()
{
    {
        SmartLock lock (_mutex);
        _started = false;
    }
    _buf_list.pause_pop ();
}

void
BufferPool::release (SmartPtr<BufferData> &data)
{
    {
        SmartLock lock (_mutex);
        if (!_started)
            return;
    }
    _buf_list.push (data);
}

bool
BufferPool::fixate_video_info (VideoBufferInfo &info)
{
    XCAM_UNUSED (info);
    return true;
}

SmartPtr<BufferProxy>
BufferPool::create_buffer_from_data (SmartPtr<BufferData> &data)
{
    const VideoBufferInfo &info = get_video_info ();

    XCAM_ASSERT (data.ptr ());
    return new BufferProxy (info, data);
}

};
