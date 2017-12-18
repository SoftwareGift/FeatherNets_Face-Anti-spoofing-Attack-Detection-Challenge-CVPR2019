/*
 * cl_memory.cpp - CL memory
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

#include "cl_utils.h"
#include "cl_memory.h"
#if HAVE_LIBDRM
#include "intel/cl_va_memory.h"
#endif

namespace XCam {

CLImageDesc::CLImageDesc ()
    : format {CL_R, CL_UNORM_INT8}
    , width (0)
    , height (0)
    , row_pitch (0)
    , slice_pitch (0)
    , array_size (0)
    , size (0)
{
}

bool
CLImageDesc::operator == (const CLImageDesc& desc) const
{
    if (desc.format.image_channel_data_type == this->format.image_channel_data_type &&
            desc.format.image_channel_order == this->format.image_channel_order &&
            desc.width == this->width &&
            desc.height == this->height &&
            desc.row_pitch == this->row_pitch &&
            desc.slice_pitch == this->slice_pitch &&
            desc.array_size == this->array_size)// &&
        //desc.size == this->size)
        return true;
    return false;
}

CLMemory::CLMemory (const SmartPtr<CLContext> &context)
    : _context (context)
    , _mem_id (NULL)
    , _mem_fd (-1)
    , _mem_need_destroy (true)
    , _mapped_ptr (NULL)
{
    XCAM_ASSERT (context.ptr () && context->is_valid ());
}

CLMemory::~CLMemory ()
{
    release_fd ();

    if (_mapped_ptr)
        enqueue_unmap (_mapped_ptr);

    if (_mem_id && _mem_need_destroy) {
        _context->destroy_mem (_mem_id);
    }
}

int32_t
CLMemory::export_fd ()
{
    if (_mem_fd >= 0)
        return _mem_fd;

#if HAVE_LIBDRM
    SmartPtr<CLIntelContext> context = _context.dynamic_cast_ptr<CLIntelContext> ();
    _mem_fd = context->export_mem_fd (_mem_id);
#endif
    if (_mem_fd < 0)
        XCAM_LOG_ERROR ("invalid fd:%d", _mem_fd);

    return _mem_fd;
}

void
CLMemory::release_fd ()
{
    if (_mem_fd <= 0)
        return;

    close (_mem_fd);
    _mem_fd = -1;
}

XCamReturn
CLMemory::enqueue_unmap (
    void *ptr,
    CLEventList &event_waits,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLContext> context = get_context ();
    cl_mem mem_id = get_mem_id ();

    XCAM_ASSERT (is_valid ());
    if (!is_valid ())
        return XCAM_RETURN_ERROR_PARAM;

    XCAM_ASSERT (ptr == _mapped_ptr);
    if (ptr == _mapped_ptr)
        _mapped_ptr = NULL;

    return context->enqueue_unmap (mem_id, ptr, event_waits, event_out);
}

bool CLMemory::get_cl_mem_info (
    cl_image_info param_name, size_t param_size,
    void *param, size_t *param_size_ret)
{
    cl_mem mem_id = get_mem_id ();
    cl_int error_code = CL_SUCCESS;
    if (!mem_id)
        return false;

    error_code = clGetMemObjectInfo (mem_id, param_name, param_size, param, param_size_ret);
    XCAM_FAIL_RETURN(
        WARNING,
        error_code == CL_SUCCESS,
        false,
        "clGetMemObjectInfo failed on param:%d, errno:%d", param_name, error_code);
    return true;
}

CLBuffer::CLBuffer (const SmartPtr<CLContext> &context)
    : CLMemory (context)
{
}

CLBuffer::CLBuffer (
    const SmartPtr<CLContext> &context, uint32_t size,
    cl_mem_flags  flags, void *host_ptr)
    : CLMemory (context)
    , _flags (flags)
    , _size (size)
{
    init_buffer (context, size, flags, host_ptr);
}

bool
CLBuffer::init_buffer (
    const SmartPtr<CLContext> &context, uint32_t size,
    cl_mem_flags  flags, void *host_ptr)
{
    cl_mem mem_id = NULL;

    mem_id = context->create_buffer (size, flags, host_ptr);
    if (mem_id == NULL) {
        XCAM_LOG_WARNING ("CLBuffer create buffer failed");
        return false;
    }

    set_mem_id (mem_id);
    return true;
}

CLSubBuffer::CLSubBuffer (
    const SmartPtr<CLContext> &context, SmartPtr<CLBuffer> main_buf,
    cl_mem_flags flags, uint32_t offset, uint32_t size)
    : CLBuffer (context)
    , _main_buf (main_buf)
    , _flags (flags)
    , _size (size)
{
    init_sub_buffer (context, main_buf, flags, offset, size);
}

bool
CLSubBuffer::init_sub_buffer (
    const SmartPtr<CLContext> &context,
    SmartPtr<CLBuffer> main_buf,
    cl_mem_flags flags,
    uint32_t offset,
    uint32_t size)
{
    cl_mem sub_mem = NULL;
    cl_mem main_mem = main_buf->get_mem_id ();
    XCAM_FAIL_RETURN (ERROR, main_mem != NULL, false, "get memory from main image failed");

    cl_buffer_region region;
    region.origin = offset;
    region.size = size;

    sub_mem = context->create_sub_buffer (main_mem, region, flags);
    if (sub_mem == NULL) {
        XCAM_LOG_WARNING ("CLBuffer create sub buffer failed");
        return false;
    }

    set_mem_id (sub_mem);
    return true;
}

XCamReturn
CLBuffer::enqueue_read (
    void *ptr, uint32_t offset, uint32_t size,
    CLEventList &event_waits,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLContext> context = get_context ();
    cl_mem mem_id = get_mem_id ();

    XCAM_ASSERT (is_valid ());
    if (!is_valid ())
        return XCAM_RETURN_ERROR_PARAM;

    return context->enqueue_read_buffer (mem_id, ptr, offset, size, true, event_waits, event_out);
}

XCamReturn
CLBuffer::enqueue_write (
    void *ptr, uint32_t offset, uint32_t size,
    CLEventList &event_waits,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLContext> context = get_context ();
    cl_mem mem_id = get_mem_id ();

    XCAM_ASSERT (is_valid ());
    if (!is_valid ())
        return XCAM_RETURN_ERROR_PARAM;

    return context->enqueue_write_buffer (mem_id, ptr, offset, size, true, event_waits, event_out);
}

XCamReturn
CLBuffer::enqueue_map (
    void *&ptr, uint32_t offset, uint32_t size,
    cl_map_flags map_flags,
    CLEventList &event_waits,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLContext> context = get_context ();
    cl_mem mem_id = get_mem_id ();
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (is_valid ());
    if (!is_valid ())
        return XCAM_RETURN_ERROR_PARAM;

    ret = context->enqueue_map_buffer (mem_id, ptr, offset, size, true, map_flags, event_waits, event_out);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "enqueue_map failed ");

    set_mapped_ptr (ptr);
    return ret;
}

CLImage::CLImage (const SmartPtr<CLContext> &context)
    : CLMemory (context)
{
}

uint32_t
CLImage::get_pixel_bytes () const
{
    return calculate_pixel_bytes(_image_desc.format);
}

bool
CLImage::get_cl_image_info (cl_image_info param_name, size_t param_size, void *param, size_t *param_size_ret)
{
    cl_mem mem_id = get_mem_id ();
    cl_int error_code = CL_SUCCESS;
    if (!mem_id)
        return false;

    error_code = clGetImageInfo (mem_id, param_name, param_size, param, param_size_ret);
    XCAM_FAIL_RETURN(
        WARNING,
        error_code == CL_SUCCESS,
        false,
        "clGetImageInfo failed on param:%d, errno:%d", param_name, error_code);
    return true;
}

uint32_t
CLImage::calculate_pixel_bytes (const cl_image_format &fmt)
{
    uint32_t a = 0, b = 0;
    switch (fmt.image_channel_order) {
    case CL_R:
    case CL_A:
    case CL_Rx:
        a = 1;
        break;
    case CL_RG:
    case CL_RA:
    case CL_RGx:
        a = 2;
        break;
    case CL_RGB:
    case CL_RGBx:
        a = 3;
        break;
    case CL_RGBA:
    case CL_BGRA:
    case CL_ARGB:
        a = 4;
        break;
    default:
        XCAM_LOG_DEBUG ("calculate_pixel_bytes with wrong channel_order:0x%04x", fmt.image_channel_order);
        return 0;
    }

    switch (fmt.image_channel_data_type) {
    case CL_UNORM_INT8:
    case CL_SNORM_INT8:
    case CL_SIGNED_INT8:
    case CL_UNSIGNED_INT8:
        b = 1;
        break;
    case CL_SNORM_INT16:
    case CL_UNORM_INT16:
    case CL_SIGNED_INT16:
    case CL_UNSIGNED_INT16:
    case CL_HALF_FLOAT:
        b = 2;
        break;
    case CL_UNORM_INT24:
        b = 3;
        break;
    case CL_SIGNED_INT32:
    case CL_UNSIGNED_INT32:
    case CL_FLOAT:
        b = 4;
        break;
    default:
        XCAM_LOG_DEBUG ("calculate_pixel_bytes with wrong channel_data_type:0x%04x", fmt.image_channel_data_type);
        return 0;
    }

    return a * b;
}

bool
CLImage::video_info_2_cl_image_desc (
    const VideoBufferInfo & video_info,
    CLImageDesc &image_desc)
{
    image_desc.width = video_info.width;
    image_desc.height = video_info.height;
    image_desc.array_size = 0;
    image_desc.row_pitch = video_info.strides[0];
    XCAM_ASSERT (image_desc.row_pitch >= image_desc.width);
    image_desc.slice_pitch = 0;

    switch (video_info.format) {
    case XCAM_PIX_FMT_RGB48:
        //cl_image_info.fmt.image_channel_order = CL_RGB;
        //cl_image_info.fmt.image_channel_data_type = CL_UNORM_INT16;
        XCAM_LOG_WARNING (
            "video_info to cl_image_info doesn't support XCAM_PIX_FMT_RGB48, maybe try XCAM_PIX_FMT_RGBA64 instread\n"
            " **** XCAM_PIX_FMT_RGB48 need check with cl implementation ****");
        return false;
        break;
    case V4L2_PIX_FMT_GREY:
        image_desc.format.image_channel_order = CL_R;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        break;

    case XCAM_PIX_FMT_RGBA64:
        image_desc.format.image_channel_order = CL_RGBA;
        image_desc.format.image_channel_data_type = CL_UNORM_INT16;
        break;

    case V4L2_PIX_FMT_RGB24:
        image_desc.format.image_channel_order = CL_RGB;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        break;

    case V4L2_PIX_FMT_RGB565:
        image_desc.format.image_channel_order = CL_RGB;
        image_desc.format.image_channel_data_type = CL_UNORM_SHORT_565;
        break;
    case V4L2_PIX_FMT_XBGR32:
    case V4L2_PIX_FMT_ABGR32:
    case V4L2_PIX_FMT_BGR32:
        image_desc.format.image_channel_order = CL_BGRA;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        break;
        // cl doesn'tn support ARGB32 up to now, how about consider V4L2_PIX_FMT_RGBA32
    case V4L2_PIX_FMT_RGB32:
    case V4L2_PIX_FMT_ARGB32:
    case V4L2_PIX_FMT_XRGB32:
        image_desc.format.image_channel_order = CL_ARGB;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        break;

    case V4L2_PIX_FMT_RGBA32:
        image_desc.format.image_channel_order = CL_RGBA;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        break;

    case V4L2_PIX_FMT_SBGGR10:
    case V4L2_PIX_FMT_SGBRG10:
    case V4L2_PIX_FMT_SGRBG10:
    case V4L2_PIX_FMT_SRGGB10:
    case V4L2_PIX_FMT_SBGGR12:
    case V4L2_PIX_FMT_SGBRG12:
    case V4L2_PIX_FMT_SGRBG12:
    case V4L2_PIX_FMT_SRGGB12:
    case V4L2_PIX_FMT_SBGGR16:
    case XCAM_PIX_FMT_SGRBG16:
        image_desc.format.image_channel_order = CL_R;
        image_desc.format.image_channel_data_type = CL_UNORM_INT16;
        break;

    case V4L2_PIX_FMT_SBGGR8:
    case V4L2_PIX_FMT_SGBRG8:
    case V4L2_PIX_FMT_SGRBG8:
    case V4L2_PIX_FMT_SRGGB8:
        image_desc.format.image_channel_order = CL_R;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        break;

    case V4L2_PIX_FMT_NV12:
        image_desc.format.image_channel_order = CL_R;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        image_desc.array_size = 2;
        image_desc.slice_pitch = video_info.strides [0] * video_info.aligned_height;
        break;

    case V4L2_PIX_FMT_YUYV:
        image_desc.format.image_channel_order = CL_RGBA;
        image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        image_desc.width /= 2;
        break;

    case XCAM_PIX_FMT_LAB:
        image_desc.format.image_channel_order = CL_R;
        image_desc.format.image_channel_data_type = CL_FLOAT;
        break;

    case XCAM_PIX_FMT_RGB48_planar:
    case XCAM_PIX_FMT_RGB24_planar:
        image_desc.format.image_channel_order = CL_RGBA;
        if (XCAM_PIX_FMT_RGB48_planar == video_info.format)
            image_desc.format.image_channel_data_type = CL_UNORM_INT16;
        else
            image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        image_desc.width = video_info.aligned_width / 4;
        image_desc.array_size = 3;
        image_desc.slice_pitch = video_info.strides [0] * video_info.aligned_height;
        break;

    case XCAM_PIX_FMT_SGRBG16_planar:
    case XCAM_PIX_FMT_SGRBG8_planar:
        image_desc.format.image_channel_order = CL_RGBA;
        if (XCAM_PIX_FMT_SGRBG16_planar == video_info.format)
            image_desc.format.image_channel_data_type = CL_UNORM_INT16;
        else
            image_desc.format.image_channel_data_type = CL_UNORM_INT8;
        image_desc.width = video_info.aligned_width / 4;
        image_desc.array_size = 4;
        image_desc.slice_pitch = video_info.strides [0] * video_info.aligned_height;
        break;

    default:
        XCAM_LOG_WARNING (
            "video_info to cl_image_info doesn't support format:%s",
            xcam_fourcc_to_string (video_info.format));
        return false;
    }

    return true;
}

void
CLImage::init_desc_by_image ()
{
    size_t width = 0, height = 0, row_pitch = 0, slice_pitch = 0, array_size = 0, mem_size = 0;
    cl_image_format format = {CL_R, CL_UNORM_INT8};

    get_cl_image_info (CL_IMAGE_FORMAT, sizeof(format), &format);
    get_cl_image_info (CL_IMAGE_WIDTH, sizeof(width), &width);
    get_cl_image_info (CL_IMAGE_HEIGHT, sizeof(height), &height);
    get_cl_image_info (CL_IMAGE_ROW_PITCH, sizeof(row_pitch), &row_pitch);
    get_cl_image_info (CL_IMAGE_SLICE_PITCH, sizeof(slice_pitch), &slice_pitch);
    get_cl_image_info (CL_IMAGE_ARRAY_SIZE, sizeof(array_size), &array_size);
    get_cl_mem_info (CL_MEM_SIZE, sizeof(mem_size), &mem_size);

    _image_desc.format = format;
    _image_desc.width = width;
    _image_desc.height = height;
    _image_desc.row_pitch = row_pitch;
    _image_desc.slice_pitch = slice_pitch;
    _image_desc.array_size = array_size;
    _image_desc.size = mem_size;
}

XCamReturn
CLImage::enqueue_map (
    void *&ptr,
    size_t *origin, size_t *region,
    size_t *row_pitch, size_t *slice_pitch,
    cl_map_flags map_flags,
    CLEventList &event_waits,
    SmartPtr<CLEvent> &event_out)
{
    SmartPtr<CLContext> context = get_context ();
    cl_mem mem_id = get_mem_id ();
    XCamReturn ret = XCAM_RETURN_NO_ERROR;

    XCAM_ASSERT (is_valid ());
    if (!is_valid ())
        return XCAM_RETURN_ERROR_PARAM;

    ret = context->enqueue_map_image (mem_id, ptr, origin, region, row_pitch, slice_pitch, true, map_flags, event_waits, event_out);
    XCAM_FAIL_RETURN (
        WARNING,
        ret == XCAM_RETURN_NO_ERROR,
        ret,
        "enqueue_map failed ");

    set_mapped_ptr (ptr);
    return ret;
}

CLImage2D::CLImage2D (
    const SmartPtr<CLContext> &context,
    const VideoBufferInfo &video_info,
    cl_mem_flags  flags)
    : CLImage (context)
{
    CLImageDesc cl_desc;

    if (!video_info_2_cl_image_desc (video_info, cl_desc)) {
        XCAM_LOG_WARNING ("CLVaImage create va image failed on default videoinfo");
        return;
    }

    init_image_2d (context, cl_desc, flags);
}

CLImage2D::CLImage2D (
    const SmartPtr<CLContext> &context,
    const CLImageDesc &cl_desc,
    cl_mem_flags  flags,
    SmartPtr<CLBuffer> bind_buf)
    : CLImage (context)
{
    _bind_buf = bind_buf;
    init_image_2d (context, cl_desc, flags);
}

bool CLImage2D::init_image_2d (
    const SmartPtr<CLContext> &context,
    const CLImageDesc &desc,
    cl_mem_flags  flags)
{
    cl_mem mem_id = 0;
    cl_image_desc cl_desc;

    xcam_mem_clear (cl_desc);
    cl_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    cl_desc.image_width = desc.width;
    cl_desc.image_height = desc.height;
    cl_desc.image_depth = 1;
    cl_desc.image_array_size = 0;
    cl_desc.image_row_pitch = 0;
    cl_desc.image_slice_pitch = 0;
    cl_desc.num_mip_levels = 0;
    cl_desc.num_samples = 0;
    cl_desc.buffer = NULL;
    if (_bind_buf.ptr ()) {
        if (desc.row_pitch)
            cl_desc.image_row_pitch = desc.row_pitch;
        else {
            cl_desc.image_row_pitch = calculate_pixel_bytes(desc.format) * desc.width;
        }
        XCAM_ASSERT (cl_desc.image_row_pitch);
        cl_desc.buffer = _bind_buf->get_mem_id ();
        XCAM_ASSERT (cl_desc.buffer);
    }

    mem_id = context->create_image (flags, desc.format, cl_desc);
    if (mem_id == NULL) {
        XCAM_LOG_WARNING ("CLImage2D create image 2d failed");
        return false;
    }
    set_mem_id (mem_id);
    init_desc_by_image ();
    return true;
}

CLImage2DArray::CLImage2DArray (
    const SmartPtr<CLContext> &context,
    const VideoBufferInfo &video_info,
    cl_mem_flags  flags,
    uint32_t extra_array_size)
    : CLImage (context)
{
    CLImageDesc cl_desc;

    XCAM_ASSERT (video_info.components >= 2);

    if (!video_info_2_cl_image_desc (video_info, cl_desc)) {
        XCAM_LOG_WARNING ("CLVaImage create va image failed on default videoinfo");
        return;
    }
    XCAM_ASSERT (cl_desc.array_size >= 2);

    //special process for BYT platform for slice-pitch
    //if (video_info.format == V4L2_PIX_FMT_NV12)
    cl_desc.height = XCAM_ALIGN_UP (cl_desc.height, 16);

    cl_desc.array_size += extra_array_size;

    init_image_2d_array (context, cl_desc, flags);
}

bool CLImage2DArray::init_image_2d_array (
    const SmartPtr<CLContext> &context,
    const CLImageDesc &desc,
    cl_mem_flags  flags)
{
    cl_mem mem_id = 0;
    cl_image_desc cl_desc;

    xcam_mem_clear (cl_desc);
    cl_desc.image_type = CL_MEM_OBJECT_IMAGE2D_ARRAY;
    cl_desc.image_width = desc.width;
    cl_desc.image_height = desc.height;
    cl_desc.image_depth = 1;
    cl_desc.image_array_size = desc.array_size;
    cl_desc.image_row_pitch = 0;
    cl_desc.image_slice_pitch = 0;
    cl_desc.num_mip_levels = 0;
    cl_desc.num_samples = 0;
    cl_desc.buffer = NULL;

    mem_id = context->create_image (flags, desc.format, cl_desc);
    if (mem_id == NULL) {
        XCAM_LOG_WARNING ("CLImage2D create image 2d failed");
        return false;
    }
    set_mem_id (mem_id);
    init_desc_by_image ();
    return true;
}


};
