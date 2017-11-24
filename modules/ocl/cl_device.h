/*
 * cl_device.h - CL device
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

#ifndef XCAM_CL_DEVICE_H
#define XCAM_CL_DEVICE_H

#include <xcam_std.h>
#include <xcam_mutex.h>
#include <CL/cl.h>

namespace XCam {

class CLContext;

struct CLDevieInfo {
    uint32_t  max_compute_unit;
    uint32_t  max_work_item_dims;
    size_t    max_work_item_sizes [3];
    size_t    max_work_group_size;
    uint32_t  image_pitch_alignment;

    CLDevieInfo ()
        : max_compute_unit (0)
        , max_work_item_dims (0)
        , max_work_group_size (0)
        , image_pitch_alignment (4)
    {
        xcam_mem_clear (max_work_item_sizes);
    }
};

// terminate () must called before program exit

class CLDevice {
public:
    ~CLDevice ();
    static SmartPtr<CLDevice> instance ();

    bool is_inited () const {
        return _inited;
    }
    const CLDevieInfo &get_device_info () {
        return _device_info;
    }
    cl_device_id get_device_id () {
        return _device_id;
    }
    cl_platform_id get_platform_id () {
        return _platform_id;
    }
    char* get_platform_name () {
        return _platform_name;
    }

    SmartPtr<CLContext> get_context ();
    void *get_extension_function (const char *func_name);
    void terminate ();

private:
    CLDevice ();
    bool init ();
    bool query_device_info (cl_device_id device_id, CLDevieInfo &info);
    bool create_default_context ();

    XCAM_DEAD_COPY (CLDevice);

private:
    static SmartPtr<CLDevice>  _instance;
    static Mutex               _instance_mutex;
    char                       _platform_name[XCAM_CL_MAX_STR_SIZE];
    cl_platform_id             _platform_id;
    cl_device_id               _device_id;
    CLDevieInfo                _device_info;
    bool                       _inited;

    //Mutex                      _context_mutex;
    SmartPtr<CLContext>        _default_context;
};

};

#endif //XCAM_CL_DEVICE_H
