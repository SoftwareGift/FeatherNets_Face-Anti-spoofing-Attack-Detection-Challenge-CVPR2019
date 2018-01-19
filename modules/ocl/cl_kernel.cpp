/*
 * cl_kernel.cpp - CL kernel
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

#include "cl_kernel.h"
#include "cl_context.h"
#include "cl_device.h"
#include "file_handle.h"

#include <sys/stat.h>

#define ENABLE_DEBUG_KERNEL 0

#define XCAM_CL_KERNEL_DEFAULT_LOCAL_WORK_SIZE 0

namespace XCam {

CLKernel::KernelMap CLKernel::_kernel_map;
Mutex CLKernel::_kernel_map_mutex;

static char*
default_cache_path () {
    static char path[XCAM_MAX_STR_SIZE] = {0};
    const char * home_dir = std::getenv ("HOME");
    if (!home_dir)
        home_dir = "/tmp";

    snprintf (
        path, XCAM_MAX_STR_SIZE - 1,
        "%s/%s", home_dir, ".xcam/");

    return path;
}

const char* CLKernel::_kernel_cache_path = default_cache_path ();

CLKernel::CLKernel (const SmartPtr<CLContext> &context, const char *name)
    : _name (NULL)
    , _kernel_id (NULL)
    , _context (context)
{
    XCAM_ASSERT (context.ptr ());
    //XCAM_ASSERT (name);

    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    set_default_work_size ();

    XCAM_OBJ_PROFILING_INIT;
}

CLKernel::~CLKernel ()
{
    destroy ();
    if (_name)
        xcam_free (_name);
}

void
CLKernel::destroy ()
{
    if (!_parent_kernel.ptr ())
        _context->destroy_kernel_id (_kernel_id);
}

static void
get_string_key_id (const char *str, uint32_t len, uint8_t key_id[8])
{
    uint32_t key[2];
    uint32_t *ptr = (uint32_t*)(str);
    uint32_t aligned_len = 0;
    uint32_t i = 0;

    xcam_mem_clear (key);
    if (!len)
        len = strlen (str);
    aligned_len = XCAM_ALIGN_DOWN (len, 8);

    for (i = 0; i < aligned_len / 8; ++i) {
        key[0] ^= ptr[0];
        key[1] ^= ptr[1];
        ptr += 2;
    }
    memcpy (key_id, key, 8);
    len -= aligned_len;
    str += aligned_len;
    for (i = 0; i < len; ++i) {
        key_id[i] ^= (uint8_t)str[i];
    }
}

XCamReturn
CLKernel::build_kernel (const XCamKernelInfo& info, const char* options)
{
    KernelMap::iterator i_kernel;
    SmartPtr<CLKernel> single_kernel;
    char key_str[1024];
    uint8_t body_key[8];
    std::string key;
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_FAIL_RETURN (ERROR, info.kernel_name, XCAM_RETURN_ERROR_PARAM, "build kernel failed since kernel name null");

    xcam_mem_clear (body_key);
    get_string_key_id (info.kernel_body, info.kernel_body_len, body_key);
    snprintf (
        key_str, sizeof(key_str),
        "%s#%02x%02x%02x%02x%02x%02x%02x%02x#%s",
        info.kernel_name,
        body_key[0], body_key[1], body_key[2], body_key[3], body_key[4], body_key[5], body_key[6], body_key[7],
        XCAM_STR(options));
    key = key_str;

    char temp_filename[XCAM_MAX_STR_SIZE] = {0};
    char cache_filename[XCAM_MAX_STR_SIZE] = {0};
    FileHandle temp_file;
    FileHandle cache_file;
    size_t read_cache_size = 0;
    size_t write_cache_size = 0;
    uint8_t *kernel_cache = NULL;
    bool load_cache = false;
    struct timeval ts;

    const char* cache_path = std::getenv ("XCAM_CL_KERNEL_CACHE_PATH");
    if (NULL == cache_path) {
        cache_path = _kernel_cache_path;
    }

    snprintf (
        cache_filename, XCAM_MAX_STR_SIZE - 1,
        "%s/%s",
        cache_path, key_str);

    {
        SmartLock locker (_kernel_map_mutex);

        i_kernel = _kernel_map.find (key);
        if (i_kernel == _kernel_map.end ()) {
            SmartPtr<CLContext>  context = get_context ();
            single_kernel = new CLKernel (context, info.kernel_name);
            XCAM_ASSERT (single_kernel.ptr ());

            if (access (cache_path, F_OK) == -1) {
                mkdir (cache_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            }

            ret = cache_file.open (cache_filename, "r");
            if (ret == XCAM_RETURN_NO_ERROR) {
                cache_file.get_file_size (read_cache_size);
                if (read_cache_size > 0) {
                    kernel_cache = (uint8_t*) xcam_malloc0 (sizeof (uint8_t) * (read_cache_size + 1));
                    if (NULL != kernel_cache) {
                        cache_file.read_file (kernel_cache, read_cache_size);
                        cache_file.close ();

                        ret = single_kernel->load_from_binary (kernel_cache, read_cache_size);
                        xcam_free (kernel_cache);
                        kernel_cache = NULL;

                        XCAM_FAIL_RETURN (
                            ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
                            "build kernel(%s) from binary failed", key_str);

                        load_cache = true;
                    }
                }
            } else {
                XCAM_LOG_DEBUG ("open kernel cache file to read failed ret(%d)", ret);
            }

            if (load_cache == false) {
                ret = single_kernel->load_from_source (info.kernel_body, strlen (info.kernel_body), &kernel_cache, &write_cache_size, options);
                XCAM_FAIL_RETURN (
                    ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
                    "build kernel(%s) from source failed", key_str);
            }

            _kernel_map.insert (std::make_pair (key, single_kernel));
            //_kernel_map[key] = single_kernel;
        } else {
            single_kernel = i_kernel->second;
        }
    }

    if (load_cache == false && NULL != kernel_cache) {
        gettimeofday (&ts, NULL);
        snprintf (
            temp_filename, XCAM_MAX_STR_SIZE - 1,
            "%s." XCAM_TIMESTAMP_FORMAT,
            cache_filename, XCAM_TIMESTAMP_ARGS (XCAM_TIMEVAL_2_USEC (ts)));

        ret = temp_file.open (temp_filename, "wb");
        if (ret == XCAM_RETURN_NO_ERROR) {
            ret = temp_file.write_file (kernel_cache, write_cache_size);
            temp_file.close ();
            if (ret == XCAM_RETURN_NO_ERROR && write_cache_size > 0) {
                rename (temp_filename, cache_filename);
            } else {
                remove (temp_filename);
            }
        } else {
            XCAM_LOG_ERROR ("open kernel cache file to write failed ret(%d)", ret);
        }
        xcam_free (kernel_cache);
        kernel_cache = NULL;
    }

    XCAM_FAIL_RETURN (
        ERROR, (single_kernel.ptr () && single_kernel->is_valid ()), XCAM_RETURN_ERROR_UNKNOWN,
        "build kernel(%s) failed, unknown error", key_str);

    ret = this->clone (single_kernel);
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, ret,
        "load kernel(%s) from kernel failed", key_str);
    return ret;
}

XCamReturn
CLKernel::load_from_source (
    const char *source, size_t length,
    uint8_t **gen_binary, size_t *binary_size,
    const char *build_option)
{
    cl_kernel new_kernel_id = NULL;

    XCAM_ASSERT (source);
    if (!source) {
        XCAM_LOG_WARNING ("kernel:%s source empty", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (_kernel_id) {
        XCAM_LOG_WARNING ("kernel:%s already build yet", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    XCAM_ASSERT (_context.ptr ());

    if (length == 0)
        length = strlen (source);

    new_kernel_id =
        _context->generate_kernel_id (
            this,
            (const uint8_t *)source, length,
            CLContext::KERNEL_BUILD_SOURCE,
            gen_binary, binary_size,
            build_option);
    XCAM_FAIL_RETURN(
        WARNING,
        new_kernel_id != NULL,
        XCAM_RETURN_ERROR_CL,
        "cl kernel(%s) load from source failed", XCAM_STR (_name));

    _kernel_id = new_kernel_id;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::load_from_binary (const uint8_t *binary, size_t length)
{
    cl_kernel new_kernel_id = NULL;

    XCAM_ASSERT (binary);
    if (!binary || !length) {
        XCAM_LOG_WARNING ("kernel:%s binary empty", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    if (_kernel_id) {
        XCAM_LOG_WARNING ("kernel:%s already build yet", XCAM_STR (_name));
        return XCAM_RETURN_ERROR_PARAM;
    }

    XCAM_ASSERT (_context.ptr ());

    new_kernel_id =
        _context->generate_kernel_id (
            this,
            binary, length,
            CLContext::KERNEL_BUILD_BINARY,
            NULL, NULL,
            NULL);
    XCAM_FAIL_RETURN(
        WARNING,
        new_kernel_id != NULL,
        XCAM_RETURN_ERROR_CL,
        "cl kernel(%s) load from binary failed", XCAM_STR (_name));

    _kernel_id = new_kernel_id;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::clone (SmartPtr<CLKernel> kernel)
{
    XCAM_FAIL_RETURN (
        WARNING,
        kernel.ptr () && kernel->is_valid (),
        XCAM_RETURN_ERROR_CL,
        "cl kernel(%s) load from kernel failed", XCAM_STR (_name));
    _kernel_id = kernel->get_kernel_id ();
    _parent_kernel = kernel;
    if (!_name && kernel->get_kernel_name ()) {
        _name = strndup (kernel->get_kernel_name (), XCAM_MAX_STR_SIZE);
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::set_arguments (const CLArgList &args, const CLWorkSize &work_size)
{
    XCamReturn ret = XCAM_RETURN_NO_ERROR;
    uint32_t i_count = 0;

    XCAM_FAIL_RETURN (
        ERROR, _arg_list.empty (), XCAM_RETURN_ERROR_PARAM,
        "cl image kernel(%s) arguments was already set, can NOT be set twice", get_kernel_name ());

    for (CLArgList::const_iterator iter = args.begin (); iter != args.end (); ++iter, ++i_count) {
        const SmartPtr<CLArgument> &arg = *iter;
        XCAM_FAIL_RETURN (
            WARNING, arg.ptr (),
            XCAM_RETURN_ERROR_PARAM, "cl image kernel(%s) argc(%d) is NULL", get_kernel_name (), i_count);

        void *adress = NULL;
        uint32_t size = 0;
        arg->get_value (adress, size);
        ret = set_argument (i_count, adress, size);
        XCAM_FAIL_RETURN (
            WARNING, ret == XCAM_RETURN_NO_ERROR,
            ret, "cl image kernel(%s) set argc(%d) failed", get_kernel_name (), i_count);
    }

    ret = set_work_size (work_size);
    XCAM_FAIL_RETURN (
        WARNING, ret == XCAM_RETURN_NO_ERROR, ret,
        "cl image kernel(%s) set worksize(global:%dx%dx%d, local:%dx%dx%d) failed",
        XCAM_STR(get_kernel_name ()),
        (int)work_size.global[0], (int)work_size.global[1], (int)work_size.global[2],
        (int)work_size.local[0], (int)work_size.local[1], (int)work_size.local[2]);

    _arg_list = args;
    return ret;
}

XCamReturn
CLKernel::set_argument (uint32_t arg_i, void *arg_addr, uint32_t arg_size)
{
    cl_int error_code = clSetKernelArg (_kernel_id, arg_i, arg_size, arg_addr);
    if (error_code != CL_SUCCESS) {
        XCAM_LOG_DEBUG ("kernel(%s) set arg_i(%d) failed", _name, arg_i);
        return XCAM_RETURN_ERROR_CL;
    }
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLKernel::set_work_size (const CLWorkSize &work_size)
{
    uint32_t i = 0;
    uint32_t work_group_size = 1;
    const CLDevieInfo &dev_info = CLDevice::instance ()->get_device_info ();

    XCAM_FAIL_RETURN (
        WARNING,
        work_size.dim <= dev_info.max_work_item_dims,
        XCAM_RETURN_ERROR_PARAM,
        "kernel(%s) work dims(%d) greater than device max dims(%d)",
        _name, work_size.dim, dev_info.max_work_item_dims);

    for (i = 0; i < work_size.dim; ++i) {
        work_group_size *= work_size.local [i];

        XCAM_FAIL_RETURN (
            WARNING,
            work_size.local [i] <= dev_info.max_work_item_sizes [i],
            XCAM_RETURN_ERROR_PARAM,
            "kernel(%s) work item(%d) size:%d is greater than device max work item size(%d)",
            _name, i, (uint32_t)work_size.local [i], (uint32_t)dev_info.max_work_item_sizes [i]);
    }

    XCAM_FAIL_RETURN (
        WARNING,
        work_group_size == 0 || work_group_size <= dev_info.max_work_group_size,
        XCAM_RETURN_ERROR_PARAM,
        "kernel(%s) work-group-size:%d is greater than device max work-group-size(%d)",
        _name, work_group_size, (uint32_t)dev_info.max_work_group_size);

    _work_size = work_size;

    return XCAM_RETURN_NO_ERROR;
}

void
CLKernel::set_default_work_size ()
{
    _work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    for (uint32_t i = 0; i < _work_size.dim; ++i) {
        //_global_work_size [i] = XCAM_CL_KERNEL_DEFAULT_GLOBAL_WORK_SIZE;
        _work_size.local [i] = XCAM_CL_KERNEL_DEFAULT_LOCAL_WORK_SIZE;
    }
}

struct KernelUserData {
    SmartPtr<CLKernel>  kernel;
    SmartPtr<CLEvent>   event;
    CLArgList           arg_list;

    KernelUserData (const SmartPtr<CLKernel> &k, SmartPtr<CLEvent> &e)
        : kernel (k)
        , event (e)
    {}
};

void
CLKernel::event_notify (cl_event event, cl_int status, void* data)
{
    KernelUserData *kernel_data = (KernelUserData *)data;
    XCAM_ASSERT (event == kernel_data->event->get_event_id ());
    XCAM_UNUSED (status);
    XCAM_UNUSED (event);

    delete kernel_data;
}

XCamReturn
CLKernel::execute (
    const SmartPtr<CLKernel> self,
    bool block,
    CLEventList &events,
    SmartPtr<CLEvent> &event_out)
{
    XCAM_ASSERT (self.ptr () == this);
    XCAM_ASSERT (_context.ptr ());
    SmartPtr<CLEvent> kernel_event = event_out;

    if (!block && !kernel_event.ptr ()) {
        kernel_event = new CLEvent;
    }

#if ENABLE_DEBUG_KERNEL
    XCAM_OBJ_PROFILING_START;
#endif

    XCamReturn ret = _context->execute_kernel (self, NULL, events, kernel_event);

    XCAM_FAIL_RETURN (
        ERROR,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "kernel(%s) execute failed", XCAM_STR(_name));


    if (block) {
        _context->finish ();
    } else {
        XCAM_ASSERT (kernel_event.ptr () && kernel_event->get_event_id ());
        KernelUserData *user_data = new KernelUserData (self, kernel_event);
        user_data->arg_list.swap (_arg_list);
        ret = _context->set_event_callback (kernel_event, CL_COMPLETE, event_notify, user_data);
        if (ret != XCAM_RETURN_NO_ERROR) {
            XCAM_LOG_WARNING ("kernel(%s) set event callback failed", XCAM_STR (_name));
            _context->finish ();
            delete user_data;
        }
    }
    _arg_list.clear ();

#if ENABLE_DEBUG_KERNEL
    _context->finish ();
    char name[1024];
    snprintf (name, 1024, "%s-%p", XCAM_STR (_name), this);
    XCAM_OBJ_PROFILING_END (name, XCAM_OBJ_DUR_FRAME_NUM);
#endif
    return ret;
}

};
