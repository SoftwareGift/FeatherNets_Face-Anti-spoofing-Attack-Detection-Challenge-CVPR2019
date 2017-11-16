/*
 * worker.cpp - worker implementation
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

#include "worker.h"

namespace XCam {

Worker::Worker (const char *name, const SmartPtr<Callback> &cb)
    : _name (NULL)
    , _callback (cb)
{
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);
}

Worker::~Worker ()
{
    xcam_mem_clear (_name);
}

bool
Worker::set_name (const char *name)
{
    XCAM_FAIL_RETURN (
        ERROR, name,
        false, "worker set name failed with parameter NULL");

    XCAM_FAIL_RETURN (
        ERROR, !_name, false,
        "worker(%s) set name(%s) failed, already got a name", XCAM_STR (get_name ()), XCAM_STR (name));

    _name = strndup (name, XCAM_MAX_STR_SIZE);
    return true;
}

bool
Worker::set_callback (const SmartPtr<Worker::Callback> &callback)
{
    XCAM_ASSERT (!_callback.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, !_callback.ptr (),
        false, "worker(%s) callback was already set", XCAM_STR(get_name ()));

    _callback = callback;
    return true;
}

void
Worker::status_check (const SmartPtr<Worker::Arguments> &args, const XCamReturn error)
{
    if (_callback.ptr ())
        _callback->work_status (this, args, error);
}

#if ENABLE_FUNC_OBJ
bool
Worker::set_func_obj (const SmartPtr<FuncObj> &obj)
{
    XCAM_FAIL_RETURN (
        ERROR, !_func_obj.ptr (),
        false, "worker(%s) func_obj was already set", XCAM_STR(get_name ()));
    _func_obj = obj;
    return true;
}

XCamReturn
Worker::work (const SmartPtr<Worker::Arguments> &args)
{
    XCamReturn ret = _func_obj->impl(args);
    status_check (args, ret);
    return ret;
}
#endif
};

namespace UnitTestWorker {
using namespace XCam;

struct UTArguments : Worker::Arguments {
    uint32_t data;
    UTArguments () : data (5) {}
};

class UnitTestWorker: public Worker {
public:
    UnitTestWorker () : Worker("UnitTestWorker") {}
    XCamReturn work (const SmartPtr<Worker::Arguments> &args) {
        SmartPtr<UTArguments> ut_args = args.dynamic_cast_ptr<UTArguments> ();
        XCAM_ASSERT (ut_args.ptr ());
        printf ("unit test worker runing on data:%d\n", ut_args->data);
        status_check (args, XCAM_RETURN_NO_ERROR);
        return XCAM_RETURN_NO_ERROR;
    }
    XCamReturn stop () {
        return XCAM_RETURN_NO_ERROR;
    }
};

class UintTestHandler {
public:
    XCamReturn work_done (
        const SmartPtr<Worker> &w, const SmartPtr<Worker::Arguments> &,
        const XCamReturn error) {
        printf ("worker(%s) done, error:%d",
                XCAM_STR(w->get_name ()), error);
        return error;
    }
};

DECLARE_WORK_CALLBACK (UTCbBridge, UintTestHandler, work_done);

void test_base_worker()
{
    SmartPtr<UintTestHandler> handler = new UintTestHandler;
    SmartPtr<Worker> worker = new UnitTestWorker;
    worker->set_callback (new UTCbBridge (handler));
    worker->work (new UTArguments);
}

};
