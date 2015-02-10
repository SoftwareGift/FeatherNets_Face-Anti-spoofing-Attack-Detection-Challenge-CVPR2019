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
#include "video_buffer.h"
#include "drm_display.h"

namespace XCam {

class DrmBoBufferPool;

class DrmBoWrapper {
    friend class DrmDisplay;
public:
    ~DrmBoWrapper ();
    drm_intel_bo *get_bo () {
        return _bo;
    }
private:
    explicit DrmBoWrapper (SmartPtr<DrmDisplay> &display, drm_intel_bo *bo);
    XCAM_DEAD_COPY (DrmBoWrapper);
private:
    SmartPtr<DrmDisplay>       _display;
    drm_intel_bo              *_bo;
};

class DrmBoBuffer
    : public VideoBuffer
{
    friend class DrmDisplay;
    friend class DrmBoBufferPool;
public:
    virtual ~DrmBoBuffer ();

    drm_intel_bo *get_bo () {
        return _bo->get_bo ();
    }

    //abstract from VideoBuffer
    virtual uint8_t *map ();
    virtual bool unmap ();

private:
    explicit DrmBoBuffer (
        SmartPtr<DrmDisplay> display,
        const VideoBufferInfo &info,
        SmartPtr<DrmBoWrapper> &bo);

    void set_parent (SmartPtr<VideoBuffer> &parent);
    void set_buf_pool (SmartPtr<DrmBoBufferPool> &buf_pool);

    XCAM_DEAD_COPY (DrmBoBuffer);

private:
    SmartPtr<DrmDisplay>       _display;
    SmartPtr<DrmBoBufferPool>  _pool;
    SmartPtr<VideoBuffer>      _parent;
    SmartPtr<DrmBoWrapper>     _bo;
};

class DrmBoBufferPool {
    friend class DrmBoBuffer;

public:
    explicit DrmBoBufferPool (SmartPtr<DrmDisplay> &display);
    ~DrmBoBufferPool ();
    bool set_buffer_info (const VideoBufferInfo &info);
    bool init (uint32_t buf_num = 6);
    void deinit ();
    SmartPtr<DrmBoBuffer> get_buffer (SmartPtr<DrmBoBufferPool> &self);

private:
    void release (SmartPtr<DrmBoWrapper> &bo);
    XCAM_DEAD_COPY (DrmBoBufferPool);

private:
    SmartPtr<DrmDisplay>     _display;
    VideoBufferInfo          _buf_info;
    SafeList<DrmBoWrapper>   _buf_list;
    uint32_t                 _buf_count;
};


};

#endif //XCAM_DRM_BO_BUFFER_H

