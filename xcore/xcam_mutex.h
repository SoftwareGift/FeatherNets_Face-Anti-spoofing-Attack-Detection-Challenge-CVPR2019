/*
 * xcam_mutex.h - Lock
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

#ifndef XCAM_MUTEX_H
#define XCAM_MUTEX_H

#include <xcam_std.h>
#include <pthread.h>
#include <sys/time.h>

namespace XCam {

class Mutex {
    friend class Cond;
private:
    XCAM_DEAD_COPY (Mutex);

public:
    Mutex () {
        int error_num = pthread_mutex_init (&_mutex, NULL);
        if (error_num != 0) {
            XCAM_LOG_WARNING ("Mutex init failed %d: %s", error_num, strerror(error_num));
        }
    }
    virtual ~Mutex () {
        int error_num = pthread_mutex_destroy (&_mutex);
        if (error_num != 0) {
            XCAM_LOG_WARNING ("Mutex destroy failed %d: %s", error_num, strerror(error_num));
        }
    }

    void lock() {
        int error_num = pthread_mutex_lock (&_mutex);
        if (error_num != 0) {
            XCAM_LOG_WARNING ("Mutex lock failed %d: %s", error_num, strerror(error_num));
        }
    }
    void unlock() {
        int error_num = pthread_mutex_unlock (&_mutex);
        if (error_num != 0) {
            XCAM_LOG_WARNING ("Mutex unlock failed %d: %s", error_num, strerror(error_num));
        }
    }

private:
    pthread_mutex_t _mutex;
};

class Cond {
private:
    XCAM_DEAD_COPY (Cond);

public:
    Cond () {
        pthread_cond_init (&_cond, NULL);
    }
    ~Cond () {
        pthread_cond_destroy (&_cond);
    }

    int wait (Mutex &mutex) {
        return pthread_cond_wait (&_cond, &mutex._mutex);
    }
    int timedwait (Mutex &mutex, uint32_t time_in_us) {
        struct timeval now;
        struct timespec abstime;

        gettimeofday (&now, NULL);
        now.tv_usec += time_in_us;
        xcam_mem_clear (abstime);
        abstime.tv_sec += now.tv_sec + now.tv_usec / 1000000;
        abstime.tv_nsec = (now.tv_usec % 1000000) * 1000;

        return pthread_cond_timedwait (&_cond, &mutex._mutex, &abstime);
    }

    int signal() {
        return pthread_cond_signal (&_cond);
    }
    int broadcast() {
        return pthread_cond_broadcast (&_cond);
    }
private:
    pthread_cond_t _cond;
};

class SmartLock {
private:
    XCAM_DEAD_COPY (SmartLock);

public:
    SmartLock (XCam::Mutex &mutex): _mutex(mutex) {
        _mutex.lock();
    }
    virtual ~SmartLock () {
        _mutex.unlock();
    }
private:
    XCam::Mutex &_mutex;
};
};
#endif //XCAM_MUTEX_H

