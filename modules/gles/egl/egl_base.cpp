/*
 * egl_base.cpp - EGL base implementation
 *
 *  Copyright (c) 2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#include "egl_base.h"

namespace XCam {

EGLBase::EGLBase ()
    : _display (EGL_NO_DISPLAY)
    , _context (EGL_NO_CONTEXT)
    , _surface (EGL_NO_SURFACE)
{
}

EGLBase::~EGLBase ()
{
    if (_display != EGL_NO_DISPLAY) {
        if (_context != EGL_NO_CONTEXT) {
            destroy_context (_display, _context);
            _context = EGL_NO_CONTEXT;
        }

        if (_surface != EGL_NO_SURFACE) {
            destroy_surface (_display, _surface);
            _surface = EGL_NO_SURFACE;
        }

        terminate (_display);
        _display = EGL_NO_DISPLAY;
    }
}

bool
EGLBase::init ()
{
    bool ret = get_display (EGL_DEFAULT_DISPLAY, _display);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: get display failed");

    EGLint major, minor;
    ret = initialize (_display, &major, &minor);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: EGL initialize failed");
    XCAM_LOG_INFO ("EGL version: %d.%d", major, minor);

    EGLConfig configs;
    EGLint num_config;
    EGLint cfg_attribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE};
    ret = choose_config (_display, cfg_attribs, &configs, 1, &num_config);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: choose config failed");

    EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    ret = create_context (_display, configs, EGL_NO_CONTEXT, ctx_attribs, _context);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: create context failed");

    ret = make_current (_display, _surface, _surface, _context);
    XCAM_FAIL_RETURN (ERROR, ret, false, "EGLInit: make current failed");

    return true;
}

bool
EGLBase::get_display (NativeDisplayType native_display, EGLDisplay &display)
{
    display = eglGetDisplay (native_display);
    XCAM_FAIL_RETURN (
        ERROR, display != EGL_NO_DISPLAY, false,
        "EGLInit: get display failed");
    return true;
}

bool
EGLBase::initialize (EGLDisplay display, EGLint *major, EGLint *minor)
{
    EGLBoolean ret = eglInitialize (display, major, minor);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: initialize failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::choose_config (
    EGLDisplay display, EGLint const *attribs, EGLConfig *configs,
    EGLint config_size, EGLint *num_config)
{
    EGLBoolean ret = eglChooseConfig (display, attribs, configs, config_size, num_config);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: choose config failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::create_context (
    EGLDisplay display, EGLConfig config, EGLContext share_context, EGLint const *attribs,
    EGLContext &context)
{
    context = eglCreateContext (display, config, share_context, attribs);
    XCAM_FAIL_RETURN (
        ERROR, context != EGL_NO_CONTEXT, false,
        "EGLInit: create context failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::create_window_surface (
    EGLDisplay display, EGLConfig config, NativeWindowType native_window, EGLint const *attribs,
    EGLSurface &surface)
{
    surface = eglCreateWindowSurface (display, config, native_window, attribs);
    XCAM_FAIL_RETURN (
        ERROR, surface != EGL_NO_SURFACE, false,
        "EGLInit: create window surface failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::make_current (EGLDisplay display, EGLSurface draw, EGLSurface read, EGLContext context)
{
    EGLBoolean ret = eglMakeCurrent (display, draw, read, context);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: make current failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::swap_buffers (EGLDisplay display, EGLSurface surface)
{
    EGLBoolean ret = eglSwapBuffers (display, surface);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: swap buffers failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    return true;
}

bool
EGLBase::destroy_context (EGLDisplay display, EGLContext &context)
{
    EGLBoolean ret = eglDestroyContext (display, context);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: destroy context failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    XCAM_FAIL_RETURN (ERROR, ret == EGL_TRUE, false, "EGLInit: destroy context failed");
    return true;
}

bool
EGLBase::destroy_surface (EGLDisplay display, EGLSurface &surface)
{
    EGLBoolean ret = eglDestroySurface (display, surface);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: destroy surface failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    XCAM_FAIL_RETURN (ERROR, ret == EGL_TRUE, false, "EGLInit: destroy surface failed");
    return true;
}

bool
EGLBase::terminate (EGLDisplay display)
{
    EGLBoolean ret = eglTerminate (display);
    XCAM_FAIL_RETURN (
        ERROR, ret == EGL_TRUE, false,
        "EGLInit: terminate failed, error flag: %s",
        EGL::error_string (EGL::get_error ()));
    XCAM_FAIL_RETURN (ERROR, ret == EGL_TRUE, false, "EGLInit: terminate failed");
    return true;
}

}
