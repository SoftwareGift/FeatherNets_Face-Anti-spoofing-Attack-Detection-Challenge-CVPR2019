/*
 * cl_context.cpp - CL context
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


#include "cl_context.h"
#include "cl_kernel.h"
#include "cl_device.h"
#include <utility>

#undef XCAM_CL_MAX_EVENT_SIZE
#define XCAM_CL_MAX_EVENT_SIZE 256

#define OCL_EXT_NAME_CREATE_BUFFER_FROM_LIBVA_INTEL "clCreateBufferFromLibvaIntel"
#define OCL_EXT_NAME_CREATE_BUFFER_FROM_FD_INTEL    "clCreateBufferFromFdINTEL"
#define OCL_EXT_NAME_CREATE_IMAGE_FROM_LIBVA_INTEL  "clCreateImageFromLibvaIntel"
#define OCL_EXT_NAME_CREATE_IMAGE_FROM_FD_INTEL     "clCreateImageFromFdINTEL"
#define OCL_EXT_NAME_GET_MEM_OBJECT_FD_INTEL        "clGetMemObjectFdIntel"

namespace XCam {

class CLKernel;

void
CLContext::context_pfn_notify (
    const char* erro_info,
    const void *private_info,
    size_t cb,
    void *user_data
)
{
    CLContext *context = (CLContext*) user_data;
    XCAM_UNUSED (context);
    XCAM_UNUSED (erro_info);
    XCAM_UNUSED (private_info);
    XCAM_UNUSED (cb);
    XCAM_LOG_DEBUG ("cl context pfn error:%s", XCAM_STR (erro_info));
}

void CLContext::program_pfn_notify (
    cl_program program, void *user_data)
{
    CLContext *context = (CLContext*) user_data;
    char kernel_names [XCAM_CL_MAX_STR_SIZE];

    XCAM_UNUSED (context);
    XCAM_UNUSED (program);
    xcam_mem_clear (kernel_names);
    //clGetProgramInfo (program, CL_PROGRAM_KERNEL_NAMES, sizeof (kernel_names) - 1, kernel_names, NULL);
    //XCAM_LOG_DEBUG ("cl program report error on kernels: %s", kernel_names);
}

uint32_t
CLContext::event_list_2_id_array (
    CLEventList &events_wait,
    cl_event *cl_events, uint32_t max_count)
{
    uint32_t num_of_events_wait = 0;

    for (CLEventList::iterator iter = events_wait.begin ();
            iter != events_wait.end (); ++iter) {
        SmartPtr<CLEvent> &event = *iter;

        if (num_of_events_wait >= max_count) {
            XCAM_LOG_WARNING ("CLEventList(%d) larger than id_array(max_count:%d)", (uint32_t)events_wait.size(), max_count);
            break;
        }
        XCAM_ASSERT (event->get_event_id ());
        cl_events[num_of_events_wait++] = event->get_event_id ();
    }

    return num_of_events_wait;
}


CLContext::CLContext (SmartPtr<CLDevice> &device)
    : _context_id (NULL)
    , _device (device)
{
    if (!init_context ()) {
        XCAM_LOG_ERROR ("CL init context failed");
    }

    XCAM_LOG_DEBUG ("CLContext constructed");
}

CLContext::~CLContext ()
{
    destroy_context ();
    XCAM_LOG_DEBUG ("CLContext destructed");
}

void
CLContext::terminate ()
{
    //_kernel_map.clear ();
    _cmd_queue_list.clear ();
}

XCamReturn
CLContext::flush ()
{
    cl_int error_code = CL_SUCCESS;
    cl_command_queue cmd_queue_id = NULL;
    SmartPtr<CLCommandQueue> cmd_queue = get_default_cmd_queue ();

    XCAM_ASSERT (cmd_queue.ptr ());
    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    error_code = clFlush (cmd_queue_id);

    XCAM_FAIL_RETURN (
        WARNING,
        error_code == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "CL flush cmdqueue failed with error_code:%d", error_code);

    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLContext::finish ()
{
    cl_int error_code = CL_SUCCESS;
    cl_command_queue cmd_queue_id = NULL;
    SmartPtr<CLCommandQueue> cmd_queue = get_default_cmd_queue ();

    XCAM_ASSERT (cmd_queue.ptr ());
    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    error_code = clFinish (cmd_queue_id);

    XCAM_FAIL_RETURN (
        WARNING,
        error_code == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "CL finish cmdqueue failed with error_code:%d", error_code);

    return XCAM_RETURN_NO_ERROR;
}

bool
CLContext::init_context ()
{
    cl_context context_id = NULL;
    cl_int err_code = 0;
    cl_device_id device_id = _device->get_device_id ();

    XCAM_ASSERT (_context_id == NULL);

    if (!_device->is_inited()) {
        XCAM_LOG_ERROR ("create cl context failed since device is not initialized");
        return false;
    }

    context_id =
        clCreateContext (NULL, 1, &device_id,
                         CLContext::context_pfn_notify, this,
                         &err_code);
    if (err_code != CL_SUCCESS)
    {
        XCAM_LOG_WARNING ("create cl context failed, error:%d", err_code);
        return false;
    }
    _context_id = context_id;
    return true;
}

bool
CLContext::init_cmd_queue (SmartPtr<CLContext> &self)
{
    XCAM_ASSERT (_cmd_queue_list.empty ());
    XCAM_ASSERT (self.ptr() == this);
    SmartPtr<CLCommandQueue> cmd_queue = create_cmd_queue (self);
    if (!cmd_queue.ptr ())
        return false;

    _cmd_queue_list.push_back (cmd_queue);
    return true;
}

SmartPtr<CLCommandQueue>
CLContext::get_default_cmd_queue ()
{
    CLCmdQueueList::iterator iter;

    XCAM_ASSERT (!_cmd_queue_list.empty ());
    if (_cmd_queue_list.empty ())
        return NULL;
    iter = _cmd_queue_list.begin ();
    return *iter;
}

void
CLContext::destroy_context ()
{
    if (!is_valid ())
        return;
    clReleaseContext (_context_id);
    _context_id = NULL;
}

XCamReturn
CLContext::execute_kernel (
    const SmartPtr<CLKernel> kernel,
    const SmartPtr<CLCommandQueue> queue,
    CLEventList &events_wait,
    SmartPtr<CLEvent> &event_out)
{
    XCAM_ASSERT (kernel.ptr ());

    cl_int error_code = CL_SUCCESS;
    cl_command_queue cmd_queue_id = NULL;
    cl_event *event_out_id = NULL;
    cl_event events_id_wait[XCAM_CL_MAX_EVENT_SIZE];
    uint32_t num_of_events_wait = 0;
    uint32_t work_group_size = 1;
    const size_t *local_sizes = NULL;
    cl_kernel kernel_id = kernel->get_kernel_id ();
    CLWorkSize work_size = kernel->get_work_size ();
    SmartPtr<CLCommandQueue> cmd_queue = queue;

    if (!cmd_queue.ptr ()) {
        cmd_queue = get_default_cmd_queue ();
    }
    XCAM_ASSERT (cmd_queue.ptr ());

    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    num_of_events_wait = event_list_2_id_array (events_wait, events_id_wait, XCAM_CL_MAX_EVENT_SIZE);
    if (event_out.ptr ())
        event_out_id = &event_out->get_event_id ();

    for (uint32_t i = 0; i < work_size.dim; ++i) {
        work_group_size *= work_size.local[i];
    }
    if (work_group_size)
        local_sizes = work_size.local;
    else
        local_sizes = NULL;

    error_code =
        clEnqueueNDRangeKernel (
            cmd_queue_id, kernel_id,
            work_size.dim, NULL, work_size.global, local_sizes,
            num_of_events_wait, (num_of_events_wait ? events_id_wait : NULL),
            event_out_id);

    XCAM_FAIL_RETURN(
        WARNING,
        error_code == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "execute kernel(%s) failed with error_code:%d",
        kernel->get_kernel_name (), error_code);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLContext::set_event_callback (
    SmartPtr<CLEvent> &event, cl_int status,
    void (*callback) (cl_event, cl_int, void*),
    void *user_data)
{
    XCAM_ASSERT (event.ptr () && event->get_event_id ());
    cl_int error_code = clSetEventCallback (event->get_event_id (), status, callback, user_data);
    return (error_code == CL_SUCCESS ? XCAM_RETURN_NO_ERROR : XCAM_RETURN_ERROR_CL);
}

SmartPtr<CLCommandQueue>
CLContext::create_cmd_queue (SmartPtr<CLContext> &self)
{
    cl_device_id device_id = _device->get_device_id ();
    cl_command_queue cmd_queue_id = NULL;
    cl_int err_code = 0;
    SmartPtr<CLCommandQueue> result;

    XCAM_ASSERT (self.ptr() == this);

#if defined (CL_VERSION_2_0) && (CL_VERSION_2_0 == 1)
    cmd_queue_id = clCreateCommandQueueWithProperties (_context_id, device_id, 0, &err_code);
#else
    cmd_queue_id = clCreateCommandQueue (_context_id, device_id, 0, &err_code);
#endif
    if (err_code != CL_SUCCESS) {
        XCAM_LOG_WARNING ("create CL command queue failed, errcode:%d", err_code);
        return NULL;
    }

    result = new CLCommandQueue (self, cmd_queue_id);
    return result;
}

cl_kernel
CLContext::generate_kernel_id (
    CLKernel *kernel,
    const uint8_t *source, size_t length,
    CLContext::KernelBuildType type,
    uint8_t **gen_binary, size_t *binary_size,
    const char *build_option)
{
    struct CLProgram {
        cl_program id;

        CLProgram ()
            : id (NULL)
        {}
        ~CLProgram () {
            if (id)
                clReleaseProgram (id);
        }
    };

    CLProgram program;
    cl_kernel kernel_id = NULL;
    cl_int error_code = CL_SUCCESS;
    cl_device_id device_id = _device->get_device_id ();
    const char * name = kernel->get_kernel_name ();

    XCAM_ASSERT (source && length);
    XCAM_ASSERT (name);

    switch (type) {
    case KERNEL_BUILD_SOURCE:
        program.id =
            clCreateProgramWithSource (
                _context_id, 1,
                (const char**)(&source), (const size_t *)&length,
                &error_code);
        break;
    case KERNEL_BUILD_BINARY:
        program.id =
            clCreateProgramWithBinary (
                _context_id, 1, &device_id,
                (const size_t *)&length, (const uint8_t**)(&source),
                NULL, &error_code);
        break;
    }

    XCAM_FAIL_RETURN (
        WARNING,
        error_code == CL_SUCCESS,
        NULL,
        "cl create program failed with error_cod:%d", error_code);
    XCAM_ASSERT (program.id);

    error_code = clBuildProgram (program.id, 1, &device_id, build_option, CLContext::program_pfn_notify, this);
    if (error_code != CL_SUCCESS) {
        //char error_log [XCAM_CL_MAX_STR_SIZE];
        char error_log [1024 * 1024 + 32];
        xcam_mem_clear (error_log);
        clGetProgramBuildInfo (program.id, device_id, CL_PROGRAM_BUILD_LOG, sizeof (error_log) - 1, error_log, NULL);
        XCAM_LOG_WARNING ("CL build program failed on %s, build log:%s", name, error_log);
        return NULL;
    }

    if (gen_binary != NULL && binary_size != NULL) {
        error_code = clGetProgramInfo (program.id, CL_PROGRAM_BINARY_SIZES, sizeof (size_t) * 1, binary_size, NULL);
        if (error_code != CL_SUCCESS) {
            XCAM_LOG_WARNING ("CL query binary sizes failed on %s, errcode:%d", name, error_code);
        }

        *gen_binary = (uint8_t *) xcam_malloc0 (sizeof (uint8_t) * (*binary_size));

        error_code = clGetProgramInfo (program.id, CL_PROGRAM_BINARIES, sizeof (uint8_t *) * 1, gen_binary, NULL);
        if (error_code != CL_SUCCESS) {
            XCAM_LOG_WARNING ("CL query program binaries failed on %s, errcode:%d", name, error_code);
        }
    }

    kernel_id = clCreateKernel (program.id, name, &error_code);
    XCAM_FAIL_RETURN (
        WARNING,
        error_code == CL_SUCCESS,
        NULL,
        "cl create kernel(%s) failed with error_cod:%d", name, error_code);

    return kernel_id;
}

void
CLContext::destroy_kernel_id (cl_kernel &kernel_id)
{
    if (kernel_id) {
        clReleaseKernel (kernel_id);
        kernel_id = NULL;
    }
}

#if 0
bool
CLContext::insert_kernel (SmartPtr<CLKernel> &kernel)
{
    std::string kernel_name = kernel->get_kernel_name ();
    CLKernelMap::iterator i_pos = _kernel_map.lower_bound (kernel_name);

    XCAM_ASSERT (!kernel_name.empty());
    if (i_pos != _kernel_map.end () && !_kernel_map.key_comp ()(kernel_name, i_pos->first)) {
        // need update
        i_pos->second = kernel;
        XCAM_LOG_DEBUG ("kernel:%s already exist in context, now update to new one", kernel_name.c_str());
        return true;
    }

    _kernel_map.insert (i_pos, std::make_pair (kernel_name, kernel));
    return true;
}
#endif

cl_mem
CLContext::create_image (
    cl_mem_flags flags, const cl_image_format& format,
    const cl_image_desc &image_info, void *host_ptr)
{
    cl_mem mem_id = NULL;
    cl_int errcode = CL_SUCCESS;

    mem_id = clCreateImage (
                 _context_id, flags,
                 &format, &image_info,
                 host_ptr, &errcode);

    XCAM_FAIL_RETURN (
        WARNING,
        errcode == CL_SUCCESS,
        NULL,
        "create cl image failed, errcode:%d", errcode);
    return mem_id;
}

void
CLContext::destroy_mem (cl_mem mem_id)
{
    if (mem_id)
        clReleaseMemObject (mem_id);
}

cl_mem
CLContext::create_buffer (uint32_t size, cl_mem_flags flags, void *host_ptr)
{
    cl_mem mem_id = NULL;
    cl_int errcode = CL_SUCCESS;

    XCAM_ASSERT (_context_id);

    mem_id = clCreateBuffer (
                 _context_id, flags,
                 size, host_ptr,
                 &errcode);

    XCAM_FAIL_RETURN (
        WARNING,
        errcode == CL_SUCCESS,
        NULL,
        "create cl buffer failed, errcode:%d", errcode);
    return mem_id;
}

cl_mem
CLContext::create_sub_buffer (
    cl_mem main_mem,
    cl_buffer_region region,
    cl_mem_flags flags)
{
    cl_mem sub_mem = NULL;
    cl_int errcode = CL_SUCCESS;

    sub_mem = clCreateSubBuffer (main_mem, flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &errcode);
    XCAM_FAIL_RETURN (
        WARNING,
        errcode == CL_SUCCESS,
        NULL,
        "create sub buffer failed, errcode:%d", errcode);

    return sub_mem;
}

XCamReturn
CLContext::enqueue_read_buffer (
    cl_mem buf_id, void *ptr,
    uint32_t offset, uint32_t size,
    bool block,
    CLEventList &events_wait,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLCommandQueue> cmd_queue;
    cl_command_queue cmd_queue_id = NULL;
    cl_event *event_out_id = NULL;
    cl_event events_id_wait[XCAM_CL_MAX_EVENT_SIZE];
    uint32_t num_of_events_wait = 0;
    cl_int errcode = CL_SUCCESS;

    cmd_queue = get_default_cmd_queue ();
    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    num_of_events_wait = event_list_2_id_array (events_wait, events_id_wait, XCAM_CL_MAX_EVENT_SIZE);
    if (event_out.ptr ())
        event_out_id = &event_out->get_event_id ();

    XCAM_ASSERT (_context_id);
    XCAM_ASSERT (cmd_queue_id);
    errcode = clEnqueueReadBuffer (
                  cmd_queue_id, buf_id,
                  (block ? CL_BLOCKING : CL_NON_BLOCKING),
                  offset, size, ptr,
                  num_of_events_wait, (num_of_events_wait ? events_id_wait : NULL),
                  event_out_id);

    XCAM_FAIL_RETURN (
        WARNING,
        errcode == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "cl enqueue read buffer failed with error_code:%d", errcode);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLContext::enqueue_write_buffer (
    cl_mem buf_id, void *ptr,
    uint32_t offset, uint32_t size,
    bool block,
    CLEventList &events_wait,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLCommandQueue> cmd_queue;
    cl_command_queue cmd_queue_id = NULL;
    cl_event *event_out_id = NULL;
    cl_event events_id_wait[XCAM_CL_MAX_EVENT_SIZE];
    uint32_t num_of_events_wait = 0;
    cl_int errcode = CL_SUCCESS;

    cmd_queue = get_default_cmd_queue ();
    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    num_of_events_wait = event_list_2_id_array (events_wait, events_id_wait, XCAM_CL_MAX_EVENT_SIZE);
    if (event_out.ptr ())
        event_out_id = &event_out->get_event_id ();

    XCAM_ASSERT (_context_id);
    XCAM_ASSERT (cmd_queue_id);
    errcode = clEnqueueWriteBuffer (
                  cmd_queue_id, buf_id,
                  (block ? CL_BLOCKING : CL_NON_BLOCKING),
                  offset, size, ptr,
                  num_of_events_wait, (num_of_events_wait ? events_id_wait : NULL),
                  event_out_id);

    XCAM_FAIL_RETURN (
        WARNING,
        errcode == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "cl enqueue write buffer failed with error_code:%d", errcode);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLContext::enqueue_map_buffer (
    cl_mem buf_id, void *&ptr,
    uint32_t offset, uint32_t size,
    bool block,
    cl_map_flags map_flags,
    CLEventList &events_wait,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLCommandQueue> cmd_queue;
    cl_command_queue cmd_queue_id = NULL;
    cl_event *event_out_id = NULL;
    cl_event events_id_wait[XCAM_CL_MAX_EVENT_SIZE];
    uint32_t num_of_events_wait = 0;
    cl_int errcode = CL_SUCCESS;
    void *out_ptr = NULL;

    cmd_queue = get_default_cmd_queue ();
    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    num_of_events_wait = event_list_2_id_array (events_wait, events_id_wait, XCAM_CL_MAX_EVENT_SIZE);
    if (event_out.ptr ())
        event_out_id = &event_out->get_event_id ();

    XCAM_ASSERT (_context_id);
    XCAM_ASSERT (cmd_queue_id);
    out_ptr = clEnqueueMapBuffer (
                  cmd_queue_id, buf_id,
                  (block ? CL_BLOCKING : CL_NON_BLOCKING),
                  map_flags,
                  offset, size,
                  num_of_events_wait, (num_of_events_wait ? events_id_wait : NULL),
                  event_out_id,
                  &errcode);

    XCAM_FAIL_RETURN (
        WARNING,
        out_ptr && errcode == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "cl enqueue map buffer failed with error_code:%d", errcode);

    ptr = out_ptr;
    return XCAM_RETURN_NO_ERROR;
}


XCamReturn
CLContext::enqueue_map_image (
    cl_mem buf_id, void *&ptr,
    const size_t *origin,
    const size_t *region,
    size_t *image_row_pitch,
    size_t *image_slice_pitch,
    bool block,
    cl_map_flags map_flags,
    CLEventList &events_wait,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLCommandQueue> cmd_queue;
    cl_command_queue cmd_queue_id = NULL;
    cl_event *event_out_id = NULL;
    cl_event events_id_wait[XCAM_CL_MAX_EVENT_SIZE];
    uint32_t num_of_events_wait = 0;
    cl_int errcode = CL_SUCCESS;
    void *out_ptr = NULL;

    cmd_queue = get_default_cmd_queue ();
    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    num_of_events_wait = event_list_2_id_array (events_wait, events_id_wait, XCAM_CL_MAX_EVENT_SIZE);
    if (event_out.ptr ())
        event_out_id = &event_out->get_event_id ();

    XCAM_ASSERT (_context_id);
    XCAM_ASSERT (cmd_queue_id);

    out_ptr = clEnqueueMapImage (
                  cmd_queue_id, buf_id,
                  (block ? CL_BLOCKING : CL_NON_BLOCKING),
                  map_flags,
                  origin,
                  region,
                  image_row_pitch,
                  image_slice_pitch,
                  num_of_events_wait, (num_of_events_wait ? events_id_wait : NULL),
                  event_out_id,
                  &errcode);

    XCAM_FAIL_RETURN (
        WARNING,
        out_ptr && errcode == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "cl enqueue map buffer failed with error_code:%d", errcode);

    ptr = out_ptr;
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLContext::enqueue_unmap (
    cl_mem mem_id,
    void *ptr,
    CLEventList &events_wait,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLCommandQueue> cmd_queue;
    cl_command_queue cmd_queue_id = NULL;
    cl_event *event_out_id = NULL;
    cl_event events_id_wait[XCAM_CL_MAX_EVENT_SIZE];
    uint32_t num_of_events_wait = 0;
    cl_int errcode = CL_SUCCESS;

    cmd_queue = get_default_cmd_queue ();
    cmd_queue_id = cmd_queue->get_cmd_queue_id ();
    num_of_events_wait = event_list_2_id_array (events_wait, events_id_wait, XCAM_CL_MAX_EVENT_SIZE);
    if (event_out.ptr ())
        event_out_id = &event_out->get_event_id ();

    XCAM_ASSERT (_context_id);
    XCAM_ASSERT (cmd_queue_id);
    errcode = clEnqueueUnmapMemObject (
                  cmd_queue_id, mem_id, ptr,
                  num_of_events_wait, (num_of_events_wait ? events_id_wait : NULL),
                  event_out_id);

    XCAM_FAIL_RETURN (
        WARNING,
        errcode == CL_SUCCESS,
        XCAM_RETURN_ERROR_CL,
        "cl enqueue unmap buffer failed with error_code:%d", errcode);

    return XCAM_RETURN_NO_ERROR;
}

CLCommandQueue::CLCommandQueue (SmartPtr<CLContext> &context, cl_command_queue id)
    : _context (context)
    , _cmd_queue_id (id)
{
    XCAM_ASSERT (context.ptr ());
    XCAM_ASSERT (id);
    XCAM_LOG_DEBUG ("CLCommandQueue constructed");
}

CLCommandQueue::~CLCommandQueue ()
{
    destroy ();
    XCAM_LOG_DEBUG ("CLCommandQueue desstructed");
}

void
CLCommandQueue::destroy ()
{
    if (_cmd_queue_id == NULL)
        return;

    clReleaseCommandQueue (_cmd_queue_id);
    _cmd_queue_id = NULL;
}

};
