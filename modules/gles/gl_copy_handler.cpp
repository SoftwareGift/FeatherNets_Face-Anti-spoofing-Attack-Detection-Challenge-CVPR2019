/*
 * gl_copy_handler.cpp - gl copy handler implementation
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

#include "gl_copy_handler.h"
#include "gl_utils.h"

#define INVALID_INDEX (uint32_t)(-1)

namespace XCam {

DECLARE_WORK_CALLBACK (CbCopyShader, GLCopyHandler, copy_shader_done);

const GLShaderInfo shader_info = {
    GL_COMPUTE_SHADER,
    "shader_copy",
#include "shader_copy.comp.slx"
    , 0
};

XCamReturn
GLCopyShader::prepare_arguments (const SmartPtr<Worker::Arguments> &base, GLCmdList &cmds)
{
    SmartPtr<GLCopyShader::Args> args = base.dynamic_cast_ptr<GLCopyShader::Args> ();
    XCAM_ASSERT (args.ptr () && args->in_buf.ptr () && args->out_buf.ptr ());

    const GLBufferDesc &in_desc = args->in_buf->get_buffer_desc ();
    const GLBufferDesc &out_desc = args->out_buf->get_buffer_desc ();
    const Rect &in_area = args->in_area;
    const Rect &out_area = args->out_area;

    XCAM_ASSERT (in_area.pos_y == 0 && out_area.pos_y == 0);
    XCAM_ASSERT (in_area.width == out_area.width && in_area.height == out_area.height);
    XCAM_ASSERT (uint32_t(in_area.height) == in_desc.height && uint32_t(out_area.height) == out_desc.height);

    cmds.push_back (new GLCmdBindBufRange (args->in_buf, 0));
    cmds.push_back (new GLCmdBindBufRange (args->out_buf, 1));

    size_t unit_bytes = 4 * sizeof (uint32_t);
    uint32_t in_img_width = XCAM_ALIGN_UP (in_desc.aligned_width, unit_bytes) / unit_bytes;
    uint32_t in_x_offset = XCAM_ALIGN_UP (in_area.pos_x, unit_bytes) / unit_bytes;
    uint32_t out_img_width = XCAM_ALIGN_UP (out_desc.aligned_width, unit_bytes) / unit_bytes;
    uint32_t out_x_offset = XCAM_ALIGN_UP (out_area.pos_x, unit_bytes) / unit_bytes;
    uint32_t copy_width = XCAM_ALIGN_UP (in_area.width, unit_bytes) / unit_bytes;
    uint32_t copy_height = XCAM_ALIGN_UP (in_area.height, 2) / 2 * 3;

    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_img_width", in_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("in_x_offset", in_x_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_img_width", out_img_width));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("out_x_offset", out_x_offset));
    cmds.push_back (new GLCmdUniformT<uint32_t> ("copy_width", copy_width));

    GLGroupsSize groups_size;
    groups_size.x = XCAM_ALIGN_UP (copy_width, 8) / 8;
    groups_size.y = XCAM_ALIGN_UP (copy_height, 8) / 8;
    groups_size.z = 1;

    SmartPtr<GLComputeProgram> prog;
    XCAM_FAIL_RETURN (
        ERROR, get_compute_program (prog), XCAM_RETURN_ERROR_PARAM,
        "GLCopyShader(%s) get compute program (idx:%d) failed", XCAM_STR (get_name ()), args->index);
    prog->set_groups_size (groups_size);
    prog->set_barrier (false);

    return XCAM_RETURN_NO_ERROR;
}

GLCopyHandler::GLCopyHandler (const char *name)
    : GLImageHandler (name)
    , _index (INVALID_INDEX)
{
}

GLCopyHandler::~GLCopyHandler ()
{
}

XCamReturn
GLCopyHandler::copy (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf)
{
    SmartPtr<ImageHandler::Parameters> param = new ImageHandler::Parameters (in_buf, out_buf);
    XCAM_ASSERT (param.ptr ());

    XCamReturn ret = execute_buffer (param, false);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GLCopyHandler(%s) copy failed", XCAM_STR (get_name ()));

    if (!out_buf.ptr ()) {
        out_buf = param->out_buf;
    }

    return ret;
}

bool
GLCopyHandler::set_copy_area (uint32_t idx, const Rect &in_area, const Rect &out_area)
{
    XCAM_FAIL_RETURN (
        ERROR,
        idx != INVALID_INDEX &&
        in_area.width == out_area.width && in_area.height == out_area.height,
        false,
        "GLCopyHandler(%s): set copy area(idx:%d) failed, input size:%dx%d output size:%dx%d", 
        XCAM_STR (get_name ()), idx, in_area.width, in_area.height, out_area.width, out_area.height);

    _index = idx;
    _in_area = in_area;
    _out_area = out_area;

    XCAM_LOG_DEBUG ("GLCopyHandler: copy area (idx:%d) input area(%d, %d, %d, %d) output area(%d, %d, %d, %d)",
        idx,
        in_area.pos_x, in_area.pos_y, in_area.width, in_area.height,
        out_area.pos_x, out_area.pos_y, out_area.width, out_area.height);

    return true;
}

XCamReturn
GLCopyHandler::configure_resource (const SmartPtr<Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr ());
    XCAM_ASSERT (!_copy_shader.ptr ());
    XCAM_FAIL_RETURN (
        ERROR,
        _index != INVALID_INDEX &&
        _in_area.width && _in_area.height && _out_area.width && _out_area.height,
        XCAM_RETURN_ERROR_PARAM,
        "GLCopyHandler(%s) invalid copy area, need set copy area first", XCAM_STR (get_name ()));

    _copy_shader = create_copy_shader ();
    XCAM_FAIL_RETURN (
        ERROR, _copy_shader.ptr (), XCAM_RETURN_ERROR_PARAM,
        "GLCopyHandler(%s) create copy shader (idx:%d) failed", XCAM_STR (get_name ()), _index);

    return XCAM_RETURN_NO_ERROR;
}

XCamReturn
GLCopyHandler::start_work (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());

    XCamReturn ret = start_copy_shader (param);
    XCAM_FAIL_RETURN (
        ERROR, xcam_ret_is_ok (ret), ret,
        "GLCopyHandler(%s) start work (idx:%d) failed", XCAM_STR (get_name ()), _index);

    param->in_buf.release ();

    return ret;
};

XCamReturn
GLCopyHandler::terminate ()
{
    if (_copy_shader.ptr ()) {
        _copy_shader.release ();
    }
    return GLImageHandler::terminate ();
}

SmartPtr<GLCopyShader>
GLCopyHandler::create_copy_shader ()
{
    SmartPtr<Worker::Callback> cb = new CbCopyShader (this);
    XCAM_ASSERT (cb.ptr ());

    SmartPtr<GLCopyShader> shader = new GLCopyShader (cb);
    XCAM_ASSERT (shader.ptr ());

    XCamReturn ret = shader->create_compute_program (shader_info, "copy_program");
    XCAM_FAIL_RETURN (
        ERROR, ret == XCAM_RETURN_NO_ERROR, NULL,
        "GLCopyHandler(%s) create compute program failed", XCAM_STR (get_name ()));

    return shader;
}

XCamReturn
GLCopyHandler::start_copy_shader (const SmartPtr<ImageHandler::Parameters> &param)
{
    XCAM_ASSERT (param.ptr () && param->in_buf.ptr () && param->out_buf.ptr ());
    XCAM_ASSERT (_copy_shader.ptr ());

    SmartPtr<GLCopyShader::Args> args = new GLCopyShader::Args (param);
    XCAM_ASSERT (args.ptr ());
    args->in_buf = get_glbuffer (param->in_buf);
    args->out_buf = get_glbuffer (param->out_buf);
    args->index = _index;
    args->in_area = _in_area;
    args->out_area = _out_area;

    return _copy_shader->work (args);
}

void
GLCopyHandler::copy_shader_done (
    const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &base, const XCamReturn error)
{
    XCAM_UNUSED (worker);
    XCAM_ASSERT (worker.ptr () == _copy_shader.ptr ());

    SmartPtr<GLCopyShader::Args> args = base.dynamic_cast_ptr<GLCopyShader::Args> ();
    XCAM_ASSERT (args.ptr ());
    const SmartPtr<ImageHandler::Parameters> param = args->get_param ();
    XCAM_ASSERT (param.ptr ());

    execute_done (param, error);
}

}
