/*
 * drm_bo_buffer.h - drm bo buffer
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */
#ifndef XCAM_DRM_BO_BUFFER_H
#define XCAM_DRM_BO_BUFFER_H

#include "xcam_utils.h"
#include "smartptr.h"
#include "safe_list.h"
#include "xcam_mutex.h"
#include "buffer_pool.h"
#include "drm_display.h"
#include "swapped_buffer.h"

namespace XCam {

class DrmBoBufferPool;
class X3aStats;

class DrmBoData
    : public BufferData
{
    friend class DrmDisplay;

public:
    ~DrmBoData ();
    drm_intel_bo *get_bo () {
        return _bo;
    }

    //derived from BufferData
    virtual uint8_t *map ();
    virtual bool unmap ();
    virtual int get_fd ();

protected:
    explicit DrmBoData (SmartPtr<DrmDisplay> &display, drm_intel_bo *bo);

    bool set_prime_fd (int fd, bool need_close);

private:
    XCAM_DEAD_COPY (DrmBoData);

private:
    SmartPtr<DrmDisplay>       _display;
    drm_intel_bo              *_bo;
    uint8_t                   *_buf;
    int                       _prime_fd;
    bool                      _need_close_fd;
};

class DrmBoBuffer
    : public virtual BufferProxy
    , public SwappedBuffer
{
    friend class DrmBoBufferPool;
    friend class DrmDisplay;

public:
    virtual ~DrmBoBuffer () {}
    drm_intel_bo *get_bo ();

    SmartPtr<X3aStats> find_3a_stats ();

protected:
    DrmBoBuffer (const VideoBufferInfo &info, const SmartPtr<DrmBoData> &data);

    //derived from SwappedBuffer
    virtual SmartPtr<SwappedBuffer> create_new_swap_buffer (
        const VideoBufferInfo &info, SmartPtr<BufferData> &data);

    XCAM_DEAD_COPY (DrmBoBuffer);
};

class DrmBoBufferPool
    : public BufferPool
{
    friend class DrmBoBuffer;

public:
    explicit DrmBoBufferPool (SmartPtr<DrmDisplay> &display);
    ~DrmBoBufferPool ();

    // **** MUST be set before set_video_info ****
    void set_swap_flags (uint32_t flags, uint32_t init_order) {
        _swap_flags = flags;
        _swap_init_order = init_order;
    }
    uint32_t get_swap_flags () const {
        return _swap_flags;
    }

    bool update_swap_init_order (uint32_t init_order);

    SmartPtr<DrmDisplay> &get_drm_display () {
        return _display;
    }

protected:
    // derived from BufferPool
    virtual bool fixate_video_info (VideoBufferInfo &info);
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

    bool init_swap_order (VideoBufferInfo &info);

private:
    XCAM_DEAD_COPY (DrmBoBufferPool);

protected:
    uint32_t                 _swap_flags;
    uint32_t                 _swap_init_order;
    uint32_t                 _swap_offsets[XCAM_VIDEO_MAX_COMPONENTS * 2];

private:
    SmartPtr<DrmDisplay>     _display;
};

};

#endif //XCAM_DRM_BO_BUFFER_H

