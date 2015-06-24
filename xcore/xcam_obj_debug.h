/*
 * xcam_obj_debug.h - object profiling and debug
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

#ifndef XCAM_OBJ_DEBUG_H
#define XCAM_OBJ_DEBUG_H

#include <stdio.h>

#if ENABLE_PROFILING
#define XCAM_OBJ_PROFILING_DEFINES                               \
    struct timeval   _profiling_start_time;                    \
    uint32_t         _profiling_times;                         \
    double           _profiling_sum_duration

#define XCAM_OBJ_PROFILING_INIT                                  \
    xcam_mem_clear (_profiling_start_time);                    \
    _profiling_times = 0;                                      \
    _profiling_sum_duration = 0.0

#define XCAM_OBJ_PROFILING_START                                 \
    gettimeofday (&_profiling_start_time, NULL)

#define XCAM_OBJ_PROFILING_END(name, times)                      \
    struct timeval profiling_now;                              \
    gettimeofday (&profiling_now, NULL);                       \
    _profiling_sum_duration +=                                 \
        (profiling_now.tv_sec - _profiling_start_time.tv_sec) * 1000.0f +  \
        (profiling_now.tv_usec - _profiling_start_time.tv_usec) / 1000.0f; \
    ++_profiling_times;                                        \
    if (_profiling_times >= times) {                           \
        char buf[1024];                                        \
        snprintf (buf, 1024, "profiling %s,average duration:%.2fms\n",     \
        (name), (_profiling_sum_duration/times));              \
        printf ("%s", buf);                                    \
        _profiling_times = 0;                                  \
        _profiling_sum_duration = 0.0;                         \
    }
#else
#define XCAM_OBJ_PROFILING_DEFINES
#define XCAM_OBJ_PROFILING_INIT
#define XCAM_OBJ_PROFILING_START
#define XCAM_OBJ_PROFILING_END(name, times)
#endif

#endif //XCAM_OBJ_DEBUG_H