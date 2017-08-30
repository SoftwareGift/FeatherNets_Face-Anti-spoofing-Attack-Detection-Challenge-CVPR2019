/*
 * xcam_common.cpp - xcam common
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

#include <base/xcam_common.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <stdarg.h>

static char log_file_name[XCAM_MAX_STR_SIZE] = {0};

void * xcam_malloc(size_t size)
{
    return malloc (size);
}

void * xcam_malloc0(size_t size)
{
    void * ptr = malloc (size);
    memset (ptr, 0, size);
    return ptr;
}

void xcam_free(void *ptr)
{
    if (ptr)
        free (ptr);
}

int xcam_device_ioctl (int fd, int cmd, void *arg)
{
    int ret = 0;
    int tried_time = 0;

    if (fd < 0)
        return -1;

    while (1) {
        ret = ioctl (fd, cmd, arg);
        if (ret >= 0)
            break;
        if (errno != EINTR && errno != EAGAIN)
            break;
        if (++tried_time > 5)
            break;
    }

    if (ret >= 0) {
        XCAM_LOG_DEBUG ("ioctl return ok on fd(%d), cmd:%d", fd, cmd);
    } else {
        XCAM_LOG_DEBUG ("ioctl failed on fd(%d), cmd:%d, error:%s",
                        fd, cmd, strerror(errno));
    }
    return ret;
}

const char *
xcam_fourcc_to_string (uint32_t fourcc)
{
    static char str[5];

    xcam_mem_clear (str);
    memcpy (str, &fourcc, 4);
    return str;
}

void xcam_print_log (const char* format, ...) {
    char buffer[XCAM_MAX_STR_SIZE] = {0};

    va_list va_list;
    va_start (va_list, format);
    vsnprintf (buffer, XCAM_MAX_STR_SIZE, format, va_list);
    va_end (va_list);

    if (strlen (log_file_name) > 0) {
        FILE* p_file = fopen (log_file_name, "ab+");
        if (NULL != p_file) {
            fwrite (buffer, sizeof (buffer[0]), strlen (buffer), p_file);
            fclose (p_file);
        } else {
            printf ("%s", buffer);
        }
    } else {
        printf ("%s", buffer);
    }
}

void xcam_set_log (const char* file_name) {
    if (NULL != file_name) {
        memset (log_file_name, 0, XCAM_MAX_STR_SIZE);
        strncpy (log_file_name, file_name, XCAM_MAX_STR_SIZE);
    }
}

