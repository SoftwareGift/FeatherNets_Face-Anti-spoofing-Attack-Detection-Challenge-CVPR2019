/*
 * xcam_common.h - xcam common and utilities
 *
 *  Copyright (c) 2014 Intel Corporation
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

#ifndef XCAM_COMMON_H
#define XCAM_COMMON_H

#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <pthread.h>
#include <math.h>
#include <inttypes.h>
#include <base/xcam_defs.h>

XCAM_BEGIN_DECLARE

typedef enum {
    XCAM_RETURN_NO_ERROR        = 0,
    XCAM_RETURN_BYPASS          = 1,

    /* errors */
    XCAM_RETURN_ERROR_PARAM     = -1,
    XCAM_RETURN_ERROR_MEM       = -2,
    XCAM_RETURN_ERROR_FILE      = -3,
    XCAM_RETURN_ERROR_AIQ       = -4,
    XCAM_RETURN_ERROR_ISP       = -5,
    XCAM_RETURN_ERROR_SENSOR    = -6,
    XCAM_RETURN_ERROR_THREAD    = -7,
    XCAM_RETURN_ERROR_IOCTL     = -8,
    XCAM_RETURN_ERROR_CL        = -9,

    XCAM_RETURN_ERROR_TIMEOUT   = -20,

    XCAM_RETURN_ERROR_UNKNOWN   = -255,
} XCamReturn;

#define xcam_malloc_type(TYPE) (TYPE*)(xcam_malloc(sizeof(TYPE)))
#define xcam_malloc0_type(TYPE) (TYPE*)(xcam_malloc0(sizeof(TYPE)))
#define xcam_mem_clear(v_stack) memset(&(v_stack), 0, sizeof(v_stack))

void * xcam_malloc (size_t size);
void * xcam_malloc0 (size_t size);

void xcam_free (void *ptr);

/*
  * return, 0 successfully
  *            else, check errno
  */
int xcam_device_ioctl (int fd, int cmd, void *arg);
const char *xcam_fourcc_to_string (uint32_t fourcc);

XCAM_END_DECLARE

#endif //XCAM_COMMON_H

