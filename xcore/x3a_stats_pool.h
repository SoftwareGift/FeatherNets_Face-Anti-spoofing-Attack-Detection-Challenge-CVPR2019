/*
 * x3a_stats_pool.h -  3a stats pool
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

#ifndef XCAM_3A_STATS_POOL_H
#define XCAM_3A_STATS_POOL_H

#include "xcam_utils.h"
#include "buffer_pool.h"
#include <base/xcam_3a_stats.h>

namespace XCam {

class X3aStatsData
    : public BufferData
{
public:
    explicit X3aStatsData (XCam3AStats *data);
    ~X3aStatsData ();
    XCam3AStats *get_stats () {
        return _data;
    }

    virtual uint8_t *map ();
    virtual bool unmap ();

private:
    XCAM_DEAD_COPY (X3aStatsData);
private:
    XCam3AStats   *_data;
};

class X3aStats
    : public BufferProxy
{
    friend class X3aStatsPool;
public:
    XCam3AStats *get_stats ();

protected:
    explicit X3aStats (const SmartPtr<X3aStatsData> &data);
    XCAM_DEAD_COPY (X3aStats);

};

class X3aStatsPool
    : public BufferPool
{
public:
    explicit X3aStatsPool ();
    XCam3AStatsInfo &get_stats_info () {
        return _stats_info;
    }
    void set_bit_depth (uint32_t bit_depth) {
        _bit_depth = bit_depth;
    }
    void set_stats_info (const XCam3AStatsInfo &info);

protected:
    virtual bool fixate_video_info (VideoBufferInfo &info);
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);

private:
    XCAM_DEAD_COPY (X3aStatsPool);

private:
    XCam3AStatsInfo    _stats_info;
    uint32_t           _bit_depth;
};

};

#endif //XCAM_3A_STATS_POOL_H

