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

namespace XCam {

class DrmBoBufferPool;

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

private:
    XCAM_DEAD_COPY (DrmBoData);
private:
    SmartPtr<DrmDisplay>       _display;
    drm_intel_bo              *_bo;
    uint8_t                   *_buf;
    int                       _prime_fd;
};

class DrmBoBuffer
    : public BufferProxy
{
    friend class DrmBoBufferPool;
    friend class DrmDisplay;
public:
    virtual ~DrmBoBuffer () {}
    drm_intel_bo *get_bo ();

protected:
    DrmBoBuffer (const VideoBufferInfo &info, const SmartPtr<DrmBoData> &data);
    XCAM_DEAD_COPY (DrmBoBuffer);
};

class DrmBoBufferPool
    : public BufferPool
{
    friend class DrmBoBuffer;

public:
    explicit DrmBoBufferPool (SmartPtr<DrmDisplay> &display);
    ~DrmBoBufferPool ();

protected:
    // derived from BufferPool
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

    SmartPtr<DrmDisplay> &get_drm_display () {
        return _display;
    }

private:
    XCAM_DEAD_COPY (DrmBoBufferPool);

private:
    SmartPtr<DrmDisplay>     _display;
};

};

#endif //XCAM_DRM_BO_BUFFER_H

