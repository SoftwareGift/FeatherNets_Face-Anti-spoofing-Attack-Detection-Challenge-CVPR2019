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

// default duration of frame numbers
#define XCAM_OBJ_DUR_FRAME_NUM 30

#define XCAM_STATIC_FPS_CALCULATION(objname, count) \
    do{                                             \
        static uint32_t num_frame = 0;              \
        static struct timeval last_sys_time;        \
        static struct timeval first_sys_time;       \
        static bool b_last_sys_time_init = false;   \
        if (!b_last_sys_time_init) {                \
            gettimeofday (&last_sys_time, NULL);    \
            gettimeofday (&first_sys_time, NULL);   \
            b_last_sys_time_init = true;            \
        } else {                                    \
            if ((num_frame%count)==0) {             \
                double total, current;              \
                struct timeval cur_sys_time;        \
                gettimeofday (&cur_sys_time, NULL); \
                total = (cur_sys_time.tv_sec - first_sys_time.tv_sec)*1.0f +     \
                    (cur_sys_time.tv_usec - first_sys_time.tv_usec)/1000000.0f;  \
                current = (cur_sys_time.tv_sec - last_sys_time.tv_sec)*1.0f +    \
                    (cur_sys_time.tv_usec - last_sys_time.tv_usec)/1000000.0f;   \
                printf("%s Current fps: %.2f, Total avg fps: %.2f\n",            \
                    #objname, ((float)(count))/current, (float)num_frame/total); \
                last_sys_time = cur_sys_time;       \
            }                                       \
        }                                           \
        ++num_frame;                                \
    }while(0)

#define XCAM_STATIC_PROFILING_START(name)               \
    static unsigned int name##_times = 0;               \
    static struct timeval name##_start_time;            \
    static struct timeval name##_end_time;              \
    gettimeofday (& name##_start_time, NULL);           \
    ++ name##_times;

#define XCAM_STATIC_PROFILING_END(name, times_of_print) \
    static double name##_sum_time = 0;                  \
    gettimeofday (& name##_end_time, NULL);             \
    name##_sum_time += (name##_end_time.tv_sec - name##_start_time.tv_sec)*1000.0f +  \
                   (name##_end_time.tv_usec - name##_start_time.tv_usec)/1000.0f; \
    if (name##_times >= times_of_print) {               \
        printf ("profiling %s, fps:%.2f duration:%.2fms\n", #name, \
            (name##_times*1000.0f/name##_sum_time), name##_sum_time/name##_times); \
        name##_times = 0;                               \
        name##_sum_time = 0.0;                          \
    }

#if ENABLE_PROFILING
#define XCAM_OBJ_PROFILING_DEFINES          \
    struct timeval   _profiling_start_time; \
    uint32_t         _profiling_times;      \
    double           _profiling_sum_duration

#define XCAM_OBJ_PROFILING_INIT             \
    xcam_mem_clear (_profiling_start_time); \
    _profiling_times = 0;                   \
    _profiling_sum_duration = 0.0

#define XCAM_OBJ_PROFILING_START \
    gettimeofday (&_profiling_start_time, NULL)

#define XCAM_OBJ_PROFILING_END(name, times) \
    struct timeval profiling_now;           \
    gettimeofday (&profiling_now, NULL);    \
    _profiling_sum_duration +=              \
        (profiling_now.tv_sec - _profiling_start_time.tv_sec) * 1000.0f +  \
        (profiling_now.tv_usec - _profiling_start_time.tv_usec) / 1000.0f; \
    ++_profiling_times;                     \
    if (_profiling_times >= times) {        \
        char buf[1024];                     \
        snprintf (buf, 1024, "profiling %s,average duration:%.2fms\n", \
        (name), (_profiling_sum_duration/times)); \
        printf ("%s", buf);                 \
        _profiling_times = 0;               \
        _profiling_sum_duration = 0.0;      \
    }
#else
#define XCAM_OBJ_PROFILING_DEFINES
#define XCAM_OBJ_PROFILING_INIT
#define XCAM_OBJ_PROFILING_START
#define XCAM_OBJ_PROFILING_END(name, times)
#endif

#endif //XCAM_OBJ_DEBUG_H
