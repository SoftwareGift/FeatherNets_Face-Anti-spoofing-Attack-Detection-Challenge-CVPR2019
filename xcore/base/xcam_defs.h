/*
 * xcam_defs.h - macros defines
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

#ifndef XCAM_DEFS_H
#define XCAM_DEFS_H

#ifndef XCAM_LOG_ERROR
#define XCAM_LOG_ERROR(format, ...)    \
    xcam_print_log ("XCAM ERROR %s:%d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif

#ifndef XCAM_LOG_WARNING
#define XCAM_LOG_WARNING(format, ...)   \
    xcam_print_log ("XCAM WARNING %s:%d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif

#ifndef XCAM_LOG_INFO
#define XCAM_LOG_INFO(format, ...)   \
    xcam_print_log ("XCAM INFO %s:%d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif

#ifdef DEBUG
#ifndef XCAM_LOG_DEBUG
#define XCAM_LOG_DEBUG(format, ...)   \
      xcam_print_log ("XCAM DEBUG %s:%d: " format "\n", __FILE__, __LINE__, ## __VA_ARGS__)
#endif
#else
#define XCAM_LOG_DEBUG(...)
#endif

#define XCAM_ASSERT(exp)  assert(exp)

#ifdef  __cplusplus
#define XCAM_BEGIN_DECLARE  extern "C" {
#define XCAM_END_DECLARE    }
#else
#define XCAM_BEGIN_DECLARE
#define XCAM_END_DECLARE
#endif

#ifndef __user
#define __user
#endif

#define XCAM_UNUSED(variable) (void)(variable)

#define XCAM_CONSTRUCTOR(obj, TYPE, ...) new (&obj) TYPE(## __VA_ARGS__)
#define XCAM_DESTRUCTOR(obj, TYPE) (obj).~TYPE()

#define XCAM_MAX(a, b)  ((a) > (b) ? (a) : (b))
#define XCAM_MIN(a, b)  ((a) < (b) ? (a) : (b))
#define XCAM_CLAMP(v, min, max)   \
    (((v) < (min)) ? (min) : (((v) > (max)) ? (max) : (v)))

#define XCAM_FAIL_RETURN(LEVEL, exp, ret, msg, ...)         \
    if (!(exp)) {                                           \
        XCAM_LOG_##LEVEL (msg, ## __VA_ARGS__);             \
        return ret;                                         \
    }

#define XCAM_RETURN_CHECK(LEVEL, exp, msg, ...)             \
    do {                                                    \
    XCamReturn err_ret = (exp);                             \
    XCAM_FAIL_RETURN(LEVEL, xcam_ret_is_ok(err_ret),        \
        err_ret, msg, ## __VA_ARGS__);                      \
    } while (0)


#define XCAM_DEAD_COPY(ClassObj)                \
        ClassObj (const ClassObj&);             \
        ClassObj & operator= (const ClassObj&)  \


#define XCAM_STR(str)  ((str) ? (str) : "null")
#define XCAM_BOOL2STR(value)  ((value) ? "true" : "false")

#define XCAM_DOUBLE_EQUAL(a, b, tolerance)  \
    (((a) >= ((b) - (tolerance))) && ((a) <= ((b) + (tolerance))))

#define XCAM_DOUBLE_EQUAL_AROUND(a, b)  \
    XCAM_DOUBLE_EQUAL((a), (b), 0.000001)

#define XCAM_GAMMA_TABLE_SIZE 256
#define XCAM_MAX_STR_SIZE 4096
#undef XCAM_CL_MAX_STR_SIZE
#define XCAM_CL_MAX_STR_SIZE 1024

#define XCAM_TIMESPEC_2_USEC(timespec) ((timespec).tv_sec*1000000LL + (timespec).tv_nsec/1000)
#define XCAM_TIMEVAL_2_USEC(timeval) ((timeval).tv_sec*1000000LL + (timeval).tv_usec)

#define XCAM_TIMESTAMP_2_SECONDS(t) ((t)/1000000)
#define XCAM_SECONDS_2_TIMESTAMP(t) ((t)*1000000)

#define XCAM_TIMESTAMP_FORMAT "%02d:%02d:%02d.%03d"

#define XCAM_TIMESTAMP_ARGS(t)                \
    (int32_t)(XCAM_TIMESTAMP_2_SECONDS(t)/3600),       \
    (int32_t)((XCAM_TIMESTAMP_2_SECONDS(t)%3600)/60),  \
    (int32_t)(XCAM_TIMESTAMP_2_SECONDS(t)%60),         \
    (int32_t)(((t)/1000)%1000)

// align must be a interger of power 2
#define XCAM_ALIGN_UP(value, align) (((value)+((align)-1))&(~((align)-1)))
#define XCAM_ALIGN_DOWN(value, align) ((value)&(~((align)-1)))
#define XCAM_ALIGN_AROUND(value, align) (((value)+(align)/2)/(align)*(align))

#ifdef _LP64
#define PRIuS "zu"
#else
#define PRIuS "u"
#endif

#ifndef XCAM_PI
#define XCAM_PI 3.1415926f
#endif

#define degree2radian(degree) ((degree) * XCAM_PI / 180.0f)

#endif //XCAM_DEFS_H
