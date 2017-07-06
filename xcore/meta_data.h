/*
 * meta_data.h - meta data struct
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_META_DATA_H
#define XCAM_META_DATA_H

#include "xcam_utils.h"
#include "smartptr.h"
#include <list>

namespace XCam {

struct MetaData
{
    int64_t timestamp; // in microseconds

    MetaData () {
        timestamp = 0;
    };
    virtual ~MetaData () {};
};

struct DevicePose
    : MetaData
{
    double   orientation[4];
    double   translation[3];
    uint32_t confidence;

    DevicePose ()
    {
        xcam_mem_clear (orientation);
        xcam_mem_clear (translation);
        confidence = 1;
    }
};

typedef std::list<SmartPtr<MetaData>>  MetaDataList;
typedef std::list<SmartPtr<DevicePose>>  DevicePoseList;

};

#endif //XCAM_META_DATA_H
