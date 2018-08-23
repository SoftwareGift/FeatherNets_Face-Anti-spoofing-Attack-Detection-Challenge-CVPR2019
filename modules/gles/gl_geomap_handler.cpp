/*
 * gl_geomap_handler.cpp - gl geometry map handler implementation
 *
 *  Copyright (c) 2018 Intel Corporation
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
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "gl_geomap_handler.h"
#include "gl_utils.h"

#define XCAM_GL_GEOMAP_ALIGN_X 4
#define XCAM_GL_GEOMAP_ALIGN_Y 2

namespace XCam {

DECLARE_WORK_CALLBACK (CbGeoMapShader, GLGeoMapHandler, geomap_shader_done);

const GLShaderInfo shader_info = {
    GL_COMPUTE_SHADER,
    "shader_geomap",
#include "shader_geomap.comp.slx"
    , 0
};

bool
GLGeoMapShader::set_std_step (float factor_x, float factor_y)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_DOUBLE_EQUAL_AROUND (factor_x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factor_y, 0.0f), false,
        "GLGeoMapShader(%s) invalid standard factors: x:%f, y:%f", XCAM_STR (get_name ()), factor_x, factor_y);

    _lut_std_step[0] = 1.0f / factor_x;
    _lut_std_step[1] = 1.0f / factor_y;

    return true;
}

XCamReturn
GLGeoMapShader::prepare_arguments (const SmartPtr<Worker::Arguments> &base, GLCmdList &cmds)
{
    SmartPtr<GLGeoMapShader::Args> args = base.dynamic_cast_ptr<GLGeoMapShader::Args> ();
    XCAM_ASSERT (args.ptr () && args->in_buf.ptr () && args->out_buf.ptr () && args->lut_buf.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        !XCAM_DOUBLE_EQUAL_AROUND (args->factors[0], 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (args->factors[1], 0.0f) &&
        !XCAM_DOUBLE_EQUAL_AROUND (args->factors[2], 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (args->factors[3], 0.0f),
        XCAM_RETURN_ERROR_PARAM,
        "GLGeoMapHandler(%s) invalid factors: %f, %f, %f, %f",
        XCAM_STR (get_name ()), args->factors[0], args->factors[1], args->factors[2], args->factors[3]);

    const GLBufferDesc &in_desc = args->in_buf->get_buffer_desc ();
    const GLBufferDesc &out_desc = args->out_buf->get_buffer_desc ();
    const GLBufferDesc &lut_desc = args->lut_buf->get_buffer_desc ();

    cmds.push_back (new GLCmdBindBufRange (args->in_buf, 0, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (args->in_buf, 1, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufRange (args->out_buf, 2, NV12PlaneYIdx));
    cmds.push_back (new GLCmdBindBufRange (args->out_buf, 3, NV12PlaneUVIdx));
    cmds.push_back (new GLCmdBindBufBase (args->lut_buf, 4));

    size_t unit_bytes = sizeof (uint32_t);
    uint32_t in_img_width = XCAM_ALIGN_UP (in_desc.width, unit_bytes) / unit_bytes;
    uint32_t out_img_width = XCAM_ALIGN_UP (out_desc.width, unit_bytes) / unit_bytes;
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_height", in_desc.height));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_height", out_desc.height));

    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_width", lut_desc.width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("lut_height", lut_desc.height));

    float lut_step[4];
    lut_step[0] = 1.0f / args->factors[0];
    lut_step[1] = 1.0f / args->factors[1];
    lut_step[2] = 1.0f / args->factors[2];
    lut_step[3] = 1.0f / args->factors[3];
    cmds.push_back (new GLCmdUniformTVect<float, 4> ("lut_step", lut_step));
    cmds.push_back (new GLCmdUniformTVect<float, 2> ("lut_std_step", _lut_std_step));

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (out_img_width, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (out_desc.height, 16) / 16;
    groups_size.z = 1;

    SmartPtr<GLComputeProgram> prog;
    XCAM_FAIL_RETURN (
        ERROR, get_compute_program (prog), XCAM_RETURN_ERROR_PARAM,
        "GLGeoMapShader(%s) get compute program failed", XCAM_STR (get_name ()));
    prog->set_groups_size (groups_size);

    return XCAM_RETURN_NO_ERROR;
}

GLGeoMapHandler::GLGeoMapHandler (const char *name)
    : GLImageHandler (name)
{
}

GLGeoMapHandler::~GLGeoMapHandler ()
{
}

XCamReturn
GLGeoMapHandler::remap (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GLGeoMapHandler(%s) remap failed", XCAM_STR (get_name ()));

    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

bool
GLGeoMapHandler::set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height)
{
    XCAM_FAIL_RETURN (
        ERROR, data && width && height, false,
        "GLGeoMapHandler(%s) set look up table failed, data ptr:%p, width:%d, height:%d",
        XCAM_STR (get_name ()), data, width, height);
    XCAM_ASSERT (!_lut_buf.ptr ());

    uint32_t lut_size = width * height * 2 * sizeof (float);
    SmartPtr<GLBuffer> buf = GLBuffer::create_buffer (GL_SHADER_STORAGE_BUFFER, NULL, lut_size);
    XCAM_ASSERT (buf.ptr ());

    GLBufferDesc desc;
    desc.width = width;
    desc.height = height;
    desc.size = lut_size;
    buf->set_buffer_desc (desc);

    float *ptr = (float *) buf->map_range (0, lut_size, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
    XCAM_FAIL_RETURN (ERROR, ptr, false, "GLGeoMapHandler(%s) map range failed", XCAM_STR (get_name ()));
    for (uint32_t i = 0; i < height; ++i) {
        float *ret = &ptr[i * width * 2];
        const PointFloat2 *line = &data[i * width];

        for (uint32_t j = 0; j < width; ++j) {
            ret[j * 2] = line[j].x;
            ret[j * 2 + 1] = line[j].y;
        }
    }
    buf->unmap ();
    _lut_buf = buf;

    return true;
}

bool
GLGeoMapHandler::init_factors ()
{
    float factor_x, factor_y;
    get_factors (factor_x, factor_y);

    if (!XCAM_DOUBLE_EQUAL_AROUND (factor_x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factor_y, 0.0f))
        return true;

    const GLBufferDesc &lut_desc = _lut_buf->get_buffer_desc ();
    return auto_calculate_factors (lut_desc.width, lut_desc.height);
}

XCamReturn
GLGeoMapHandler::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());
    XCAM_FAIL_RETURN (
        ERROR, _lut_buf.ptr (), XCAM_RETURN_ERROR_PARAM,
        "GLGeoMapHandler(%s) configure failed, look up table is empty", XCAM_STR (get_name ()));

    const VideoBufferInfo &in_info = param->in_buf->get_video_info ();
    XCAM_FAIL_RETURN (
        ERROR, in_info.format == V4L2_PIX_FMT_NV12, XCAM_RETURN_ERROR_PARAM,
        "GLGeoMapHandler(%s) only support NV12 format, but input format is %s",
        XCAM_STR (get_name ()), xcam_fourcc_to_string (in_info.format));

    uint32_t width, height;
    get_output_size (width, height);
    VideoBufferInfo out_info;
    out_info.init (
        in_info.format, width, height,
        XCAM_ALIGN_UP (width, XCAM_GL_GEOMAP_ALIGN_X),
        XCAM_ALIGN_UP (height, XCAM_GL_GEOMAP_ALIGN_Y));
    set_out_video_info (out_info);

    init_factors ();

    XCAM_ASSERT (!_geomap_shader.ptr ());
    _geomap_shader = create_geomap_shader ();
    XCAM_FAIL_RETURN (
        ERROR, _geomap_shader.ptr (), XCAM_RETURN_ERROR_PARAM,
        "GLGeoMapHandler(%s) create geomap shader failed", XCAM_STR (get_name ()));

    float factor_x, factor_y;
    get_factors (factor_x, factor_y);
    _geomap_shader->set_std_step (factor_x, factor_y);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLGeoMapHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());

    XCamReturn ret = start_geomap_shader (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GLGeoMapHandler(%s) start work failed", XCAM_STR (get_name ()));

    param->in_buf.release ();

    return ret;
};

XCamReturn
GLGeoMapHandler::terminate ()
{
    if (_geomap_shader.ptr ()) {
        _geomap_shader.release ();
    }

    return GLImageHandler::terminate ();
}

SmartPtr<GLGeoMapShader>
GLGeoMapHandler::create_geomap_shader ()
{
    SmartPtr<Worker::Callback> cb = new CbGeoMapShader (this);
    XCAM_ASSERT (cb.ptr ());
    SmartPtr<GLGeoMapShader> shader = new GLGeoMapShader (cb);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shader_info, "geomap_program");
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, NULL,
        "GLGeoMapHandler(%s) create compute program failed", XCAM_STR (get_name ()));

    return shader;
}

XCamReturn
GLGeoMapHandler::start_geomap_shader (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());
    XCAM_ASSERT (_geomap_shader.ptr ());
    XCAM_ASSERT (_lut_buf.ptr ());

    float factor_x, factor_y;
    get_factors (factor_x, factor_y);

    SmartPtr<GLGeoMapShader::Args> args = new GLGeoMapShader::Args (param);
    XCAM_ASSERT (args.ptr ());
    args->in_buf = get_glbuffer (param->in_buf);
    args->out_buf = get_glbuffer (param->out_buf);
    args->lut_buf = _lut_buf;
    args->factors[0] = factor_x;
    args->factors[1] = factor_y;
    args->factors[2] = args->factors[0];
    args->factors[3] = args->factors[1];

    return _geomap_shader->work (args);
}

void
GLGeoMapHandler::geomap_shader_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (worker.ptr () == _geomap_shader.ptr ());

    SmartPtr<GLGeoMapShader::Args> args = base.dynamic_cast_ptr<GLGeoMapShader::Args> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    execute_done (param, error);
}

GLDualConstGeoMapHandler::GLDualConstGeoMapHandler (const char *name)
    : GLGeoMapHandler (name)
    , _left_factor_x (0.0f)
    , _left_factor_y (0.0f)
    , _right_factor_x (0.0f)
    , _right_factor_y (0.0f)
{
}

GLDualConstGeoMapHandler::~GLDualConstGeoMapHandler ()
{
}

bool
GLDualConstGeoMapHandler::set_left_factors (float x, float y)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f), false,
        "GLGeoMapHandler(%s) set factors failed: x:%f, y:%f", XCAM_STR (get_name ()), x, y);

    _left_factor_x = x;
    _left_factor_y = y;

    return true;
}

bool
GLDualConstGeoMapHandler::set_right_factors (float x, float y)
{
    XCAM_FAIL_RETURN (
        ERROR, !XCAM_DOUBLE_EQUAL_AROUND (x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (y, 0.0f), false,
        "GLGeoMapHandler(%s) set factors failed: x:%f, y:%f", XCAM_STR (get_name ()), x, y);

    _right_factor_x = x;
    _right_factor_y = y;

    return true;
}

bool
GLDualConstGeoMapHandler::init_factors ()
{
    float factor_x, factor_y;
    get_factors (factor_x, factor_y);

    if (!XCAM_DOUBLE_EQUAL_AROUND (factor_x, 0.0f) && !XCAM_DOUBLE_EQUAL_AROUND (factor_y, 0.0f))
        return true;

    const GLBufferDesc &lut_desc = _lut_buf->get_buffer_desc ();
    XCAM_FAIL_RETURN (
        ERROR, auto_calculate_factors (lut_desc.width, lut_desc.height), false,
        "GLGeoMapHandler(%s) auto calculate factors failed");

    get_factors (factor_x, factor_y);
    _left_factor_x = factor_x;
    _left_factor_y = factor_y;
    _right_factor_x = _left_factor_x;
    _right_factor_y = _left_factor_y;

    return true;
}

XCamReturn
GLDualConstGeoMapHandler::start_geomap_shader (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());
    XCAM_ASSERT (_geomap_shader.ptr ());
    XCAM_ASSERT (_lut_buf.ptr ());

    SmartPtr<GLGeoMapShader::Args> args = new GLGeoMapShader::Args (param);
    XCAM_ASSERT (args.ptr ());
    args->in_buf = get_glbuffer (param->in_buf);
    args->out_buf = get_glbuffer (param->out_buf);
    args->lut_buf = _lut_buf;

    float factor_x, factor_y;
    get_left_factors (factor_x, factor_y);
    args->factors[0] = factor_x;
    args->factors[1] = factor_y;

    get_right_factors (factor_x, factor_y);
    args->factors[2] = factor_x;
    args->factors[3] = factor_y;

    return _geomap_shader->work (args);
}

SmartPtr<GLImageHandler> create_gl_geo_mapper ()
{
    SmartPtr<GLImageHandler> mapper = new GLGeoMapHandler ();
    XCAM_ASSERT (mapper.ptr ());

    return mapper;
}

SmartPtr<GeoMapper>
GeoMapper::create_gl_geo_mapper ()
{
    SmartPtr<GLImageHandler> handler = XCam::create_gl_geo_mapper ();
    return handler.dynamic_cast_ptr<GeoMapper> ();
}

}
