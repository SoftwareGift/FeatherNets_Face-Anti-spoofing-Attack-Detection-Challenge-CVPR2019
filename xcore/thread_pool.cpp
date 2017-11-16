/*
 * thread_pool.cpp - Thread Pool
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 */

#include "thread_pool.h"

#define XCAM_POOL_MIN_THREADS 2
#define XCAM_POOL_MAX_THREADS 1024

namespace XCam {

class UserThread
    : public Thread
{
public:
    UserThread (const SmartPtr<ThreadPool> &pool, const char *name)
        : Thread (name)
        , _pool (pool)
    {}

protected:
    virtual bool started ();
    virtual void stopped ();
    virtual bool loop ();

private:
    SmartPtr<ThreadPool> _pool;
};

bool
UserThread::started ()
{
    XCAM_ASSERT (_pool.ptr ());
    SmartLock lock (_pool->_mutex);
    return true;
}

void
UserThread::stopped ()
{
    XCAM_LOG_DEBUG ("thread(%s, %p) stopped", XCAM_STR(get_name ()), this);
}

bool
UserThread::loop ()
{
    XCAM_ASSERT (_pool.ptr ());
    {
        SmartLock lock (_pool->_mutex);
        if (!_pool->_running)
            return false;
    }

    SmartPtr<ThreadPool::UserData> data = _pool->_data_queue.pop ();
    if (!data.ptr ()) {
        XCAM_LOG_DEBUG ("user thread(%s) get null data, need stop", XCAM_STR (_pool->get_name ()));
        return false;
    }

    {
        SmartLock lock (_pool->_mutex);
        XCAM_ASSERT (_pool->_free_threads > 0);
        --_pool->_free_threads;
    }

    bool ret = _pool->dispatch (data);

    if (ret) {
        SmartLock lock (_pool->_mutex);
        ++_pool->_free_threads;
    }
    return ret;
}

bool
ThreadPool::dispatch (const SmartPtr<ThreadPool::UserData> &data)
{
    XCAM_FAIL_RETURN (
        ERROR, data.ptr(), true,
        "ThreadPool(%s) dispatch NULL data", XCAM_STR (get_name ()));
    XCamReturn err = data->run ();
    data->done (err);
    return true;
}

ThreadPool::ThreadPool (const char *name)
    : _name (NULL)
    , _min_threads (XCAM_POOL_MIN_THREADS)
    , _max_threads (XCAM_POOL_MIN_THREADS)
    , _allocated_threads (0)
    , _free_threads (0)
    , _running (false)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);
}

ThreadPool::~ThreadPool ()
{
    stop ();

    xcam_mem_clear (_name);
}

bool
ThreadPool::set_threads (uint32_t min, uint32_t max)
{
    XCAM_FAIL_RETURN (
        ERROR, !_running, false,
        "ThreadPool(%s) set threads failed, need stop the pool first", XCAM_STR(get_name ()));

    if (min < XCAM_POOL_MIN_THREADS)
        min = XCAM_POOL_MIN_THREADS;
    if (max > XCAM_POOL_MAX_THREADS)
        max = XCAM_POOL_MAX_THREADS;

    if (min > max)
        min = max;

    _min_threads = min;
    _max_threads = max;
    return true;
}

bool
ThreadPool::is_running ()
{
    SmartLock locker(_mutex);
    return _running;
}

XCamReturn
ThreadPool::start ()
{
    SmartLock locker(_mutex);
    if (_running)
        return XCAM_RETURN_NO_ERROR;

    _free_threads = 0;
    _allocated_threads = 0;
    _data_queue.resume_pop ();

    for (uint32_t i = 0; i < _min_threads; ++i) {
        XCamReturn ret = create_user_thread_unsafe ();
        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (ret), ret,
            "thread pool(%s) start failed by creating user thread", XCAM_STR (get_name()));
    }

    XCAM_ASSERT (_allocated_threads == _min_threads);

    _running = true;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ThreadPool::stop ()
{
    UserThreadList threads;
    {
        SmartLock locker(_mutex);
        if (!_running)
            return XCAM_RETURN_NO_ERROR;

        _running = false;
        threads = _thread_list;
        _thread_list.clear ();
    }

    for (UserThreadList::iterator i = threads.begin (); i != threads.end (); ++i)
    {
        SmartPtr<UserThread> t = *i;
        XCAM_ASSERT (t.ptr ());
        t->emit_stop ();
    }

    _data_queue.pause_pop ();
    _data_queue.clear ();

    for (UserThreadList::iterator i = threads.begin (); i != threads.end (); ++i)
    {
        SmartPtr<UserThread> t = *i;
        XCAM_ASSERT (t.ptr ());
        t->stop ();
    }

    {
        SmartLock locker(_mutex);
        _free_threads = 0;
        _allocated_threads = 0;
    }

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ThreadPool::create_user_thread_unsafe ()
{
    char name[256];
    snprintf (name, 255, "%s-%d", XCAM_STR (get_name()), _allocated_threads);
    SmartPtr<UserThread> thread = new UserThread (this, name);
    XCAM_ASSERT (thread.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, thread.ptr () && thread->start (), XCAM_RETURN_ERROR_THREAD,
        "ThreadPool(%s) create user thread failed by starting error", XCAM_STR (get_name()));

    _thread_list.push_back (thread);

    ++_allocated_threads;
    ++_free_threads;
    XCAM_ASSERT (_free_threads <= _allocated_threads);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
ThreadPool::queue (const SmartPtr<UserData> &data)
{
    XCAM_ASSERT (data.ptr ());
    {
        SmartLock locker (_mutex);
        if (!_running)
            return XCAM_RETURN_ERROR_THREAD;
    }

    if (!_data_queue.push (data))
        return XCAM_RETURN_ERROR_THREAD;

    do {
        SmartLock locker(_mutex);
        if (!_running) {
            _data_queue.erase (data);
            return XCAM_RETURN_ERROR_THREAD;
        }

        if (_allocated_threads >= _max_threads)
            break;

        if (!_free_threads)
            break;

        XCamReturn err = create_user_thread_unsafe ();
        if (!xcam_ret_is_ok (err) && _allocated_threads) {
            XCAM_LOG_WARNING ("thread pool(%s) create new thread failed but queue data can continue");
            break;
        }

        XCAM_FAIL_RETURN (
            ERROR, xcam_ret_is_ok (err), err,
            "thread pool(%s) queue data failed by creating user thread", XCAM_STR (get_name()));

    } while (0);

    return XCAM_RETURN_NO_ERROR;
}

}
