/*
 * test_utils.h - test utils
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
 * Author: John Ye <john.ye@intel.com>
 */

#ifndef XCAM_TEST_COMMON_H
#define XCAM_TEST_COMMON_H

#undef CHECK_DECLARE
#undef CHECK
#undef CHECK_CONTINUE

#define CHECK_DECLARE(level, exp, statement, msg, ...) \
    if (!(exp)) {        \
        XCAM_LOG_##level (msg, ## __VA_ARGS__);   \
        statement;                              \
    }

#define CHECK(ret, msg, ...)  \
    CHECK_DECLARE(ERROR, (ret) == XCAM_RETURN_NO_ERROR, return -1, msg, ## __VA_ARGS__)

#define CHECK_CONTINUE(ret, msg, ...)  \
    CHECK_DECLARE(WARNING, (ret) == XCAM_RETURN_NO_ERROR, , msg, ## __VA_ARGS__)

#define CHECK_EXP(exp, msg, ...) \
    CHECK_DECLARE(ERROR, exp, return -1, msg, ## __VA_ARGS__)

#define CAPTURE_DEVICE_VIDEO "/dev/video3"
#define CAPTURE_DEVICE_STILL "/dev/video0"
#define DEFAULT_CAPTURE_DEVICE CAPTURE_DEVICE_VIDEO

#define DEFAULT_EVENT_DEVICE   "/dev/v4l-subdev6"
#define DEFAULT_CPF_FILE       "/etc/atomisp/imx185.cpf"
#define DEFAULT_SAVE_FILE_NAME "capture_buffer"
#define DEFAULT_DYNAMIC_3A_LIB "/usr/lib/xcam/libxcam_3a_aiq.so"
#define DEFAULT_HYBRID_3A_LIB "/usr/lib/xcam/libxcam_3a_hybrid.so"
#define DEFAULT_SMART_ANALYSIS_LIB_DIR "/usr/lib/xcam/smartlib"


#define FPS_CALCULATION(objname, count)                     \
    do{                                              \
        static uint32_t num_frame = 0;                  \
        static struct timeval last_sys_time;         \
        static struct timeval first_sys_time;        \
        static bool b_last_sys_time_init = false;     \
        if (!b_last_sys_time_init) {                 \
          gettimeofday (&last_sys_time, NULL);       \
          gettimeofday (&first_sys_time, NULL);      \
          b_last_sys_time_init = true;               \
        } else {                                     \
          if ((num_frame%count)==0) {                   \
            double total, current;                   \
            struct timeval cur_sys_time;             \
            gettimeofday (&cur_sys_time, NULL);      \
            total = (cur_sys_time.tv_sec - first_sys_time.tv_sec)*1.0f +       \
                   (cur_sys_time.tv_usec - first_sys_time.tv_usec)/1000000.0f; \
            current = (cur_sys_time.tv_sec - last_sys_time.tv_sec)*1.0f +      \
                    (cur_sys_time.tv_usec - last_sys_time.tv_usec)/1000000.0f; \
            printf("%s Current fps: %.2f, Total avg fps: %.2f\n",              \
                    #objname, ((float)(count))/current, (float)num_frame/total);   \
            last_sys_time = cur_sys_time;            \
          }                                          \
        }                                            \
        ++num_frame;                                 \
    }while(0)


#define PROFILING_START(name) \
    static unsigned int name##_times = 0;                   \
    static struct timeval name##_start_time;         \
    static struct timeval name##_end_time;           \
    gettimeofday (& name##_start_time, NULL);        \
    ++ name##_times;

#define PROFILING_END(name, times_of_print) \
    static double name##_sum_time = 0;        \
    gettimeofday (& name##_end_time, NULL); \
    name##_sum_time += (name##_end_time.tv_sec - name##_start_time.tv_sec)*1000.0f +  \
                   (name##_end_time.tv_usec - name##_start_time.tv_usec)/1000.0f;      \
    if (name##_times >= times_of_print) {                  \
        printf ("profiling %s, fps:%.2f duration:%.2fms\n", #name, (name##_times*1000.0f/name##_sum_time), name##_sum_time/name##_times); \
        name##_times = 0;                   \
        name##_sum_time = 0.0;       \
    }

#endif  // XCAM_TEST_COMMON_H
