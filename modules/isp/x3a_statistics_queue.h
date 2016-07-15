/*
 * x3a_statistics_queue.h - statistics queue
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

#ifndef XCAM_3A_STATISTIC_QUEUE_H
#define XCAM_3A_STATISTIC_QUEUE_H

#include "xcam_utils.h"
#include "xcam_mutex.h"
#include "smartptr.h"
#include "x3a_stats_pool.h"
#include <linux/atomisp.h>

namespace XCam {

class X3aStatisticsQueue;

class X3aIspStatsData
    : public X3aStatsData
{
public:
    explicit X3aIspStatsData (struct atomisp_3a_statistics *isp_data, XCam3AStats *data);
    ~X3aIspStatsData ();
    struct atomisp_3a_statistics *get_isp_stats () {
        return _isp_data;
    }

    virtual uint8_t *map () {
        return (uint8_t*)(void*)(_isp_data);
    }
    virtual bool unmap () {
        return true;
    }

    bool fill_standard_stats ();

private:
    XCAM_DEAD_COPY (X3aIspStatsData);

private:
    struct atomisp_3a_statistics *_isp_data;
};

class X3aIspStatistics
    : public X3aStats
{
    friend class X3aStatisticsQueue;
protected:
    explicit X3aIspStatistics (const SmartPtr<X3aIspStatsData> &stats_data);

public:
    virtual ~X3aIspStatistics ();
    struct atomisp_3a_statistics *get_isp_stats ();

    bool fill_standard_stats ();

private:
    XCAM_DEAD_COPY (X3aIspStatistics);
};

class X3aStatisticsQueue
    : public X3aStatsPool
{
public:
    explicit X3aStatisticsQueue ();
    ~X3aStatisticsQueue();

    void set_grid_info (const struct atomisp_grid_info &info);

protected:
    virtual bool fixate_video_info (VideoBufferInfo &info);
    virtual SmartPtr<BufferData> allocate_data (const VideoBufferInfo &buffer_info);
    virtual SmartPtr<BufferProxy> create_buffer_from_data (SmartPtr<BufferData> &data);


private:
    struct atomisp_3a_statistics *alloc_isp_statsictics ();
    XCAM_DEAD_COPY (X3aStatisticsQueue);

private:
    struct atomisp_grid_info  _grid_info;
};

};

#endif //XCAM_3A_STATISTIC_QUEUE_H
