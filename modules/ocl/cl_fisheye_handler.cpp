/*
 * cl_fisheye_handler.cpp - CL fisheye handler
 *
 *  Copyright (c) 2016 Intel Corporation
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
#include "cl_fisheye_handler.h"
#include "cl_device.h"

#define XCAM_LSC_ARRAY_SIZE 64

static const float max_gray_threshold = 220.0f;
static const float min_gray_threshold = 80.0f;

static const float lsc_array[XCAM_LSC_ARRAY_SIZE] = {
    1.000000f, 1.000150f, 1.000334f, 1.000523f, 1.000761f, 1.001317f, 1.002109f, 1.003472f,
    1.004502f, 1.008459f, 1.011816f, 1.014686f, 1.016767f, 1.018425f, 1.020455f, 1.022125f,
    1.023080f, 1.025468f, 1.029810f, 1.035422f, 1.041943f, 1.047689f, 1.054206f, 1.059395f,
    1.063541f, 1.068729f, 1.074158f, 1.082766f, 1.088606f, 1.095224f, 1.102773f, 1.112865f,
    1.117108f, 1.132849f, 1.140659f, 1.147847f, 1.157544f, 1.165002f, 1.175248f, 1.181730f,
    1.196203f, 1.205452f, 1.216974f, 1.236338f, 1.251963f, 1.269212f, 1.293479f, 1.311051f,
    1.336007f, 1.357711f, 1.385124f, 1.409937f, 1.448611f, 1.473716f, 1.501837f, 1.525721f,
    1.555186f, 1.602372f, 1.632105f, 1.698443f, 1.759641f, 1.836303f, 1.939085f, 2.066358f
};

namespace XCam {

#define DEFAULT_FISHEYE_TABLE_SCALE 8.0f

enum {
    KernelFisheye2GPS,
    KernelFisheyeTable,
    KernelLSCTable
};

const XCamKernelInfo kernel_fisheye_info[] = {
    {
        "kernel_fisheye_2_gps",
#include "kernel_fisheye.clx"
        , 0,
    },
    {
        "kernel_fisheye_table",
#include "kernel_fisheye.clx"
        , 0,
    },
    {
        "kernel_lsc_table",
#include "kernel_fisheye.clx"
        , 0,
    },
};

CLFisheye2GPSKernel::CLFisheye2GPSKernel (
    const SmartPtr<CLContext> &context, SmartPtr<CLFisheyeHandler> &handler)
    : CLImageKernel (context)
    , _handler (handler)
{
    XCAM_ASSERT (handler.ptr ());
}

XCamReturn
CLFisheye2GPSKernel::prepare_arguments (CLArgList &args, CLWorkSize &work_size)
{
    SmartPtr<CLImage> input_y = _handler->get_input_image (NV12PlaneYIdx);
    SmartPtr<CLImage> input_uv = _handler->get_input_image (NV12PlaneUVIdx);
    SmartPtr<CLImage> output_y = _handler->get_output_image (NV12PlaneYIdx);
    SmartPtr<CLImage> output_uv = _handler->get_output_image (NV12PlaneUVIdx);
    const CLImageDesc &input_y_desc = input_y->get_image_desc ();
    const CLImageDesc &outuv_desc = output_uv->get_image_desc ();
    FisheyeInfo fisheye_info;
    float input_y_size[2];
    float out_center[2]; //width/height
    float radian_per_pixel[2];

    input_y_size[0] = input_y_desc.width;
    input_y_size[1] = input_y_desc.height;

    uint32_t dst_w, dst_h;
    float dst_range_x, dst_range_y;
    _handler->get_output_size (dst_w, dst_h);
    out_center[0] = (float)dst_w / 2.0f;
    out_center[1] = (float)dst_h / 2.0f;

    _handler->get_dst_range (dst_range_x, dst_range_y);
    radian_per_pixel[0] = degree2radian (dst_range_x) / (float)dst_w;
    radian_per_pixel[1] = degree2radian (dst_range_y) / (float)dst_h;

    fisheye_info = _handler->get_fisheye_info ();
    fisheye_info.wide_angle = degree2radian (fisheye_info.wide_angle);
    fisheye_info.rotate_angle = degree2radian (fisheye_info.rotate_angle);

    XCAM_LOG_DEBUG ("@CLFisheye2GPSKernel input size(%d, %d), out_center:(%d, %d), range:(%d,%d)",
                    (int)input_y_size[0], (int)input_y_size[1],
                    (int)out_center[0], (int)out_center[1],
                    (int)dst_range_x, (int)dst_range_y);

    args.push_back (new CLMemArgument (input_y));
    args.push_back (new CLMemArgument (input_uv));
    args.push_back (new CLArgumentTArray<float, 2> (input_y_size));
    args.push_back (new CLArgumentT<FisheyeInfo> (fisheye_info));
    args.push_back (new CLMemArgument (output_y));
    args.push_back (new CLMemArgument (output_uv));
    args.push_back (new CLArgumentTArray<float, 2> (out_center));
    args.push_back (new CLArgumentTArray<float, 2> (radian_per_pixel));

    work_size.dim = XCAM_DEFAULT_IMAGE_DIM;
    work_size.local[0] = 16;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (outuv_desc.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (outuv_desc.height, work_size.local[1]);

    return XCAM_RETURN_NO_ERROR;
}

CLFisheyeHandler::CLFisheyeHandler (const SmartPtr<CLContext> &context, SurroundMode surround_mode, bool use_map, bool need_lsc, bool need_scale)
    : CLImageHandler (context, "CLFisheyeHandler")
    , _output_width (0)
    , _output_height (0)
    , _range_longitude (180.0f)
    , _range_latitude (180.0f)
    , _map_factor (DEFAULT_FISHEYE_TABLE_SCALE)
    , _use_map (use_map)
    , _need_lsc (need_lsc ? 1 : 0)
    , _need_scale (need_scale ? 1 : 0)
    , _lsc_array_size (0)
    , _lsc_array (NULL)
    , _stable_y_start (0.0f)
    , _left_scale_factor (1.0f, 1.0f)
    , _right_scale_factor (1.0f, 1.0f)
    , _surround_mode (surround_mode)
{
    xcam_mem_clear (_gray_threshold);
}

CLFisheyeHandler::~CLFisheyeHandler()
{
    if (_lsc_array)
        xcam_free (_lsc_array);
}

void
CLFisheyeHandler::set_output_size (uint32_t width, uint32_t height)
{
    _output_width = width;
    _output_height = height;
}

void
CLFisheyeHandler::get_output_size (uint32_t &width, uint32_t &height) const
{
    width = _output_width;
    height = _output_height;
}

void
CLFisheyeHandler::set_dst_range (float longitude, float latitude)
{
    _range_longitude = longitude;
    _range_latitude = latitude;
}

void
CLFisheyeHandler::get_dst_range (float &longitude, float &latitude) const
{
    longitude = _range_longitude;
    latitude = _range_latitude;
}

void
CLFisheyeHandler::set_fisheye_info (const FisheyeInfo &info)
{
    _fisheye_info = info;
}

void
CLFisheyeHandler::set_lsc_table (float *table, uint32_t table_size)
{
    if (_lsc_array)
        xcam_free (_lsc_array);

    _lsc_array_size = table_size;
    _lsc_array = (float *) xcam_malloc0 (_lsc_array_size * sizeof (float));
    XCAM_ASSERT (_lsc_array);
    memcpy (_lsc_array, table, _lsc_array_size * sizeof (float));
}

void
CLFisheyeHandler::set_lsc_gray_threshold (float min_threshold, float max_threshold)
{
    _gray_threshold[0] = min_threshold;
    _gray_threshold[1] = max_threshold;
}

void
CLFisheyeHandler::set_stable_y_start (float y_start)
{
    _stable_y_start = y_start;
}

void
CLFisheyeHandler::set_left_scale_factor (PointFloat2 factor)
{
    _left_scale_factor = factor;
}

void
CLFisheyeHandler::set_right_scale_factor (PointFloat2 factor)
{
    _right_scale_factor = factor;
}

XCamReturn
CLFisheyeHandler::prepare_buffer_pool_video_info (
    const VideoBufferInfo &input,
    VideoBufferInfo &output)
{
    XCAM_FAIL_RETURN (
        WARNING, input.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "CLFisheyeHandler(%s) input buffer format(%s) is not supported, try NV12",
        get_name (), xcam_fourcc_to_string (input.format));

    if (!_output_width || !_output_height) {
        return XCAM_RETURN_ERROR_PARAM;
    }
    XCAM_FAIL_RETURN (
        WARNING, _output_width && _output_height, XCAM_RETURN_ERROR_PARAM,
        "CLFisheyeHandler output size(%d, %d) should > 0",
        _output_width, _output_height);

    output.init (
        input.format, _output_width, _output_height,
        XCAM_ALIGN_UP (_output_width, 16), XCAM_ALIGN_UP (_output_height, 16));
    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLFisheyeHandler::prepare_parameters (SmartPtr<VideoBuffer> &input, SmartPtr<VideoBuffer> &output)
{
    const VideoBufferInfo &in_info = input->get_video_info ();
    const VideoBufferInfo &out_info = output->get_video_info ();
    SmartPtr<CLContext> context = get_context ();
    uint32_t input_image_w = XCAM_ALIGN_DOWN (in_info.width, 2);
    uint32_t input_image_h = XCAM_ALIGN_DOWN (in_info.height, 2);

    XCAM_FAIL_RETURN (
        WARNING, _fisheye_info.is_valid (), XCAM_RETURN_ERROR_PARAM,
        "CLFisheyeHandler fisheye info is not valid, please check");

    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_R;
    cl_desc.width = input_image_w;
    cl_desc.height = input_image_h;
    cl_desc.row_pitch = in_info.strides[NV12PlaneYIdx];
    _input[NV12PlaneYIdx] = convert_to_climage (context, input, cl_desc, in_info.offsets[NV12PlaneYIdx]);

    cl_desc.format.image_channel_data_type = CL_UNORM_INT8;
    cl_desc.format.image_channel_order = CL_RG;
    cl_desc.width = input_image_w / 2;
    cl_desc.height = input_image_h / 2;
    cl_desc.row_pitch = in_info.strides[NV12PlaneUVIdx];
    _input[NV12PlaneUVIdx] = convert_to_climage (context, input, cl_desc, in_info.offsets[NV12PlaneUVIdx]);

    if (_use_map) {
        cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT16;
        cl_desc.format.image_channel_order = CL_RGBA;
        cl_desc.width = XCAM_ALIGN_DOWN (out_info.width, 8) / 8; //CL_RGBA * CL_UNSIGNED_INT16 = 8
        cl_desc.height = XCAM_ALIGN_DOWN (out_info.height, 2);
        cl_desc.row_pitch = out_info.strides[NV12PlaneYIdx];
        _output[NV12PlaneYIdx] = convert_to_climage (context, output, cl_desc, out_info.offsets[NV12PlaneYIdx]);
        cl_desc.height /= 2;
        cl_desc.row_pitch = out_info.strides[NV12PlaneUVIdx];
        _output[NV12PlaneUVIdx] = convert_to_climage (context, output, cl_desc, out_info.offsets[NV12PlaneUVIdx]);
    } else {
        cl_desc.format.image_channel_data_type = CL_UNSIGNED_INT8;
        cl_desc.format.image_channel_order = CL_RGBA;
        cl_desc.width = XCAM_ALIGN_DOWN (out_info.width, 4) / 4; //CL_RGBA * CL_UNSIGNED_INT8 = 4
        cl_desc.height = XCAM_ALIGN_DOWN (out_info.height, 2);
        cl_desc.row_pitch = out_info.strides[NV12PlaneYIdx];
        _output[NV12PlaneYIdx] = convert_to_climage (context, output, cl_desc, out_info.offsets[NV12PlaneYIdx]);
        cl_desc.height /= 2;
        cl_desc.row_pitch = out_info.strides[NV12PlaneUVIdx];
        _output[NV12PlaneUVIdx] = convert_to_climage (context, output, cl_desc, out_info.offsets[NV12PlaneUVIdx]);
    }

    XCAM_ASSERT (
        _input[NV12PlaneYIdx].ptr () && _input[NV12PlaneYIdx]->is_valid () &&
        _input[NV12PlaneUVIdx].ptr () && _input[NV12PlaneUVIdx]->is_valid () &&
        _output[NV12PlaneYIdx].ptr () && _output[NV12PlaneYIdx]->is_valid () &&
        _output[NV12PlaneUVIdx].ptr () && _output[NV12PlaneUVIdx]->is_valid ());

    if (_use_map && !_geo_table.ptr ()) {
        generate_fisheye_table (input_image_w, input_image_h, _fisheye_info);
    }

    if (!_lsc_table.ptr () && _need_lsc)
        generate_lsc_table (input_image_w, input_image_h, _fisheye_info);

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImage>
CLFisheyeHandler::create_cl_image (
    uint32_t width, uint32_t height, cl_channel_order order, cl_channel_type type)
{
    CLImageDesc cl_desc;
    cl_desc.format.image_channel_data_type = type;
    cl_desc.format.image_channel_order = order;
    cl_desc.width = width;
    cl_desc.height = height;

    SmartPtr<CLContext> context = get_context ();
    XCAM_ASSERT (context.ptr ());
    SmartPtr<CLImage> image = new CLImage2D (context, cl_desc);
    XCAM_FAIL_RETURN (
        ERROR, image.ptr () && image->is_valid (),
        NULL, "[%s] create cl image failed", get_name ());
    return image;
}

#if 0
static void
dump_geo_table (SmartPtr<CLImage> table)
{
    const CLImageDesc &desc = table->get_image_desc ();
    void *ptr = NULL;
    size_t origin[3] = {0, 0, 0};
    size_t region[3] = {desc.width, desc.height, 1};
    size_t row_pitch;
    size_t slice_pitch;

    char name[1024];
    snprintf (name, 1024, "geo_table_x_%dx%d.x", desc.width, desc.height);
    FILE *fp = fopen (name, "wb");
    XCamReturn ret = table->enqueue_map (ptr, origin, region, &row_pitch, &slice_pitch, CL_MAP_READ);
    XCAM_ASSERT (ret == XCAM_RETURN_NO_ERROR);

    for (uint32_t i = 0; i < desc.height; ++i) {
        float * line = (float*)((uint8_t*)ptr + row_pitch * i);
        for (uint32_t j = 0; j < desc.width; ++j) {
            float *buf = line + j * 4;
            if (i == 120)
                printf ("%.02f,", *buf);
            uint8_t val = *buf * 255;
            fwrite (&val, sizeof (val), 1, fp);
        }
    }
    printf ("\n");
    fclose (fp);
    table->enqueue_unmap (ptr);
}
#endif

XCamReturn
CLFisheyeHandler::generate_fisheye_table (
    uint32_t fisheye_width, uint32_t fisheye_height, const FisheyeInfo &fisheye_info)
{
    SmartPtr<CLContext> context = get_context ();
    XCAM_ASSERT (context.ptr ());
    SmartPtr<CLKernel> table_kernel = new CLKernel (context, "fisheye_table_temp");
    XCAM_FAIL_RETURN (
        ERROR, table_kernel->build_kernel (kernel_fisheye_info[KernelFisheyeTable], NULL) == XCAM_RETURN_NO_ERROR,
        XCAM_RETURN_ERROR_CL, "[%s] build fisheye table kernel failed", get_name ());

    float longitude, latitude;
    get_dst_range (longitude, latitude);
    XCAM_FAIL_RETURN (
        ERROR, longitude > 0.0f && latitude > 0.0f,
        XCAM_RETURN_ERROR_PARAM, "[%s] dest latitude and longitude were not set", get_name ());

    uint32_t output_width, output_height;
    get_output_size (output_width, output_height);

    uint32_t table_width, table_height;
    table_width = output_width / _map_factor;
    table_width = XCAM_ALIGN_UP (table_width, 4);
    table_height = output_height / _map_factor;
    table_height = XCAM_ALIGN_UP (table_height, 2);
    _geo_table = create_cl_image (table_width, table_height, CL_RGBA, CL_FLOAT);
    XCAM_FAIL_RETURN (
        ERROR, _geo_table.ptr () && _geo_table->is_valid (),
        XCAM_RETURN_ERROR_MEM, "[%s] check geo map buffer failed", get_name ());

    if(_surround_mode == BowlView) {
        BowlDataConfig bowl_data_config = get_bowl_config();
        IntrinsicParameter intrinsic_param = get_intrinsic_param();
        ExtrinsicParameter extrinsic_param = get_extrinsic_param();

        SurViewFisheyeDewarp::MapTable map_table(table_width * table_height * 2);
        PolyFisheyeDewarp fd;
        fd.set_intrinsic_param(intrinsic_param);
        fd.set_extrinsic_param(extrinsic_param);

        fd.fisheye_dewarp(map_table, table_width, table_height, output_width, output_height, bowl_data_config);

        float *map_ptr = NULL;
        size_t origin[3] = {0, 0, 0};
        size_t region[3] = {table_width, table_height, 1};
        size_t row_pitch;
        size_t slice_pitch;
        XCamReturn ret = _geo_table->enqueue_map ((void *&)map_ptr, origin, region, &row_pitch, &slice_pitch, CL_MAP_WRITE);
        XCAM_FAIL_RETURN (ERROR, xcam_ret_is_ok (ret), ret, "CLFisheyeHandler mesh table failed in enqueue_map");

        for (uint32_t row = 0; row < table_height; row++) {
            for(uint32_t col = 0; col < table_width; col++) {
                map_ptr[row * row_pitch / 4 + col * 4] = map_table[row * table_width + col].x / fisheye_width;
                map_ptr[row * row_pitch / 4 + col * 4 + 1] = map_table[row * table_width + col].y / fisheye_height;
            }
        }
        _geo_table->enqueue_unmap ((void *&)map_ptr);
    } else {
        CLArgList args;
        CLWorkSize work_size;

        FisheyeInfo fisheye_arg1 = fisheye_info;
        fisheye_arg1.wide_angle = degree2radian (fisheye_info.wide_angle);
        fisheye_arg1.rotate_angle = degree2radian (fisheye_info.rotate_angle);
        args.push_back (new CLArgumentT<FisheyeInfo> (fisheye_arg1));

        float fisheye_image_size[2];
        fisheye_image_size[0] = fisheye_width;
        fisheye_image_size[1] = fisheye_height;
        args.push_back (new CLArgumentTArray<float, 2> (fisheye_image_size));
        args.push_back (new CLMemArgument (_geo_table));

        float radian_per_pixel[2];
        radian_per_pixel[0] = degree2radian (longitude / table_width);
        radian_per_pixel[1] = degree2radian (latitude / table_height);
        args.push_back (new CLArgumentTArray<float, 2> (radian_per_pixel));

        float table_center[2];
        table_center[0] = table_width / 2.0f;
        table_center[1] = table_height / 2.0f;
        args.push_back (new CLArgumentTArray<float, 2> (table_center));

        work_size.dim = 2;
        work_size.local[0] = 8;
        work_size.local[1] = 4;
        work_size.global[0] = XCAM_ALIGN_UP (table_width, work_size.local[0]);
        work_size.global[1] = XCAM_ALIGN_UP (table_height, work_size.local[1]);

        XCAM_FAIL_RETURN (
            ERROR, table_kernel->set_arguments (args, work_size) == XCAM_RETURN_NO_ERROR,
            XCAM_RETURN_ERROR_CL, "kernel_fisheye_table set arguments failed");

        XCAM_FAIL_RETURN (
            ERROR, table_kernel->execute (table_kernel, true) == XCAM_RETURN_NO_ERROR,
            XCAM_RETURN_ERROR_CL, "[%s] execute kernel_fisheye_table failed", get_name ());
    }
    //dump_geo_table (_geo_table);

    return XCAM_RETURN_NO_ERROR;
}

void
CLFisheyeHandler::ensure_lsc_params ()
{
    if (_lsc_array)
        return;

    _lsc_array_size = XCAM_LSC_ARRAY_SIZE;
    _lsc_array = (float *) xcam_malloc0 (_lsc_array_size * sizeof (float));
    XCAM_ASSERT (_lsc_array);
    memcpy (_lsc_array, lsc_array, _lsc_array_size * sizeof (float));

    _gray_threshold[1] = max_gray_threshold;
    _gray_threshold[0] = min_gray_threshold;
}

XCamReturn
CLFisheyeHandler::generate_lsc_table (
    uint32_t fisheye_width, uint32_t fisheye_height, FisheyeInfo &fisheye_info)
{
    if (!_need_lsc) {
        XCAM_LOG_WARNING ("lsc is not needed, don't generate lsc table");
        return XCAM_RETURN_NO_ERROR;
    }

    if (!_geo_table.ptr ()) {
        XCAM_LOG_ERROR ("generate lsc table failed, need generate fisheye table first");
        return XCAM_RETURN_ERROR_MEM;
    }

    ensure_lsc_params ();

    SmartPtr<CLContext> context = get_context ();
    XCAM_ASSERT (context.ptr ());
    SmartPtr<CLKernel> table_kernel = new CLKernel (context, "lsc_table");
    XCAM_FAIL_RETURN (
        ERROR, table_kernel->build_kernel (kernel_fisheye_info[KernelLSCTable], NULL) == XCAM_RETURN_NO_ERROR,
        XCAM_RETURN_ERROR_CL, "[%s] build lsc table kernel failed", get_name ());

    SmartPtr<CLBuffer> array_buf = new CLBuffer (
        context, _lsc_array_size * sizeof (float),
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, _lsc_array);
    xcam_free (_lsc_array);

    CLImageDesc desc = _geo_table->get_image_desc ();
    _lsc_table = create_cl_image (desc.width, desc.height, CL_R, CL_FLOAT);
    XCAM_FAIL_RETURN (
        ERROR, _lsc_table.ptr () && _lsc_table->is_valid (),
        XCAM_RETURN_ERROR_MEM, "[%s] create lsc image failed", get_name ());

    CLArgList args;
    args.push_back (new CLMemArgument (_geo_table));
    args.push_back (new CLMemArgument (_lsc_table));
    args.push_back (new CLMemArgument (array_buf));
    args.push_back (new CLArgumentT<uint32_t> (_lsc_array_size));
    args.push_back (new CLArgumentT<FisheyeInfo> (fisheye_info));

    float fisheye_image_size[2];
    fisheye_image_size[0] = fisheye_width;
    fisheye_image_size[1] = fisheye_height;
    args.push_back (new CLArgumentTArray<float, 2> (fisheye_image_size));

    CLWorkSize work_size;
    work_size.dim = 2;
    work_size.local[0] = 8;
    work_size.local[1] = 4;
    work_size.global[0] = XCAM_ALIGN_UP (desc.width, work_size.local[0]);
    work_size.global[1] = XCAM_ALIGN_UP (desc.height, work_size.local[1]);

    XCAM_FAIL_RETURN (
        ERROR, table_kernel->set_arguments (args, work_size) == XCAM_RETURN_NO_ERROR,
        XCAM_RETURN_ERROR_CL, "kernel_lsc_table set arguments failed");

    XCAM_FAIL_RETURN (
        ERROR, table_kernel->execute (table_kernel, true) == XCAM_RETURN_NO_ERROR,
        XCAM_RETURN_ERROR_CL, "[%s] execute kernel_lsc_table failed", get_name ());

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
CLFisheyeHandler::execute_done (SmartPtr<VideoBuffer> &output)
{
    XCAM_UNUSED (output);

    for (int i = 0; i < NV12PlaneMax; ++i) {
        _input[i].release ();
        _output[i].release ();
    }

    return XCAM_RETURN_NO_ERROR;
}

SmartPtr<CLImage>
CLFisheyeHandler::get_geo_input_image (NV12PlaneIdx index) {
    return get_input_image(index);
}

SmartPtr<CLImage>
CLFisheyeHandler::get_geo_output_image (NV12PlaneIdx index) {
    return get_output_image (index);
}

void
CLFisheyeHandler::get_geo_equivalent_out_size (float &width, float &height)
{
    width = _output_width;
    height = _output_height;
}

void
CLFisheyeHandler::get_geo_pixel_out_size (float &width, float &height)
{
    width = _output_width;
    height = _output_height;
}

SmartPtr<CLImage>
CLFisheyeHandler::get_lsc_table () {
    XCAM_ASSERT (_lsc_table.ptr ());
    return _lsc_table;
}

float*
CLFisheyeHandler::get_lsc_gray_threshold () {
    return _gray_threshold;
}

static SmartPtr<CLImageKernel>
create_fishey_gps_kernel (const SmartPtr<CLContext> &context, SmartPtr<CLFisheyeHandler> handler)
{
    SmartPtr<CLImageKernel> kernel = new CLFisheye2GPSKernel (context, handler);
    XCAM_ASSERT (kernel.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, kernel->build_kernel (kernel_fisheye_info[KernelFisheye2GPS], NULL) == XCAM_RETURN_NO_ERROR,
        NULL, "build fisheye kernel failed");
    return kernel;
}

SmartPtr<CLImageHandler>
create_fisheye_handler (const SmartPtr<CLContext> &context, SurroundMode surround_mode, bool use_map, bool need_lsc, bool need_scale)
{
    SmartPtr<CLFisheyeHandler> handler;
    SmartPtr<CLImageKernel> kernel;

    handler = new CLFisheyeHandler (context, surround_mode, use_map, need_lsc, need_scale);
    XCAM_ASSERT (handler.ptr ());

    if (use_map) {
        kernel = create_geo_map_kernel (context, handler, need_lsc, need_scale);
    } else {
        kernel = create_fishey_gps_kernel (context, handler);
    }
    XCAM_FAIL_RETURN (
        ERROR, kernel.ptr (), NULL, "Fisheye handler create kernel failed.");

    handler->add_kernel (kernel);
    return handler;
}


}
