/*
 * egl_base.h - EGL base class
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

#ifndef XCAM_EGL_BASE_H
#define XCAM_EGL_BASE_H

#include <gles/egl/egl_utils.h>

namespace XCam {

class EGLBase {
public:
    explicit EGLBase ();
    ~EGLBase ();

    bool init ();

    bool get_display (NativeDisplayType native_display, EGLDisplay &display);
    bool initialize (EGLDisplay display, EGLint *major, EGLint *minor);
    bool choose_config (
        EGLDisplay display, EGLint const *attribs, EGLConfig *configs,
        EGLint config_size, EGLint *num_config);
    bool create_context (
        EGLDisplay display, EGLConfig config, EGLContext share_context, EGLint const *attribs,
        EGLContext &context);
    bool create_window_surface (
        EGLDisplay display, EGLConfig config, NativeWindowType native_window, EGLint const *attribs,
        EGLSurface &surface);
    bool make_current (EGLDisplay display, EGLSurface draw, EGLSurface read, EGLContext context);
    bool swap_buffers (EGLDisplay display, EGLSurface surface);

    bool destroy_context (EGLDisplay display, EGLContext &context);
    bool destroy_surface (EGLDisplay display, EGLSurface &surface);
    bool terminate (EGLDisplay display);

private:
    EGLDisplay        _display;
    EGLContext        _context;
    EGLSurface        _surface;
};

}

#endif // XCAM_EGL_BASE_H
