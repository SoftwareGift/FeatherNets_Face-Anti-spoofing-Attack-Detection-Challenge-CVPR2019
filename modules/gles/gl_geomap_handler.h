/*
 * gl_geomap_handler.h - gl geometry map handler class
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

#ifndef XCAM_GL_GEOMAP_HANDER_H
#define XCAM_GL_GEOMAP_HANDER_H

#include <interface/geo_mapper.h>
#include <gles/gl_image_shader.h>
#include <gles/gl_image_handler.h>

namespace XCam {

class GLGeoMapShader
    : public GLImageShader
{
public:
    struct Args : GLArgs {
        SmartPtr<GLBuffer>        in_buf, out_buf;
        SmartPtr<GLBuffer>        lut_buf;
        float                     factors[4];

        Args (const SmartPtr<ImageHandler::Parameters> &param)
            : GLArgs (param)
        {}
    };

public:
    explicit GLGeoMapShader (const SmartPtr<Worker::Callback> &cb)
        : GLImageShader ("GLGeoMapShader", cb)
    {
        xcam_mem_clear (_lut_std_step);
    }

    ~GLGeoMapShader () {}
    bool set_std_step (float factor_x, float factor_y);

private:
    virtual XCamReturn prepare_arguments (const SmartPtr<Worker::Arguments> &args, GLCmdList &cmds);
    XCAM_DEAD_COPY (GLGeoMapShader);

private:
    float        _lut_std_step[2];
};

class GLGeoMapHandler
    : public GLImageHandler, public GeoMapper
{
    friend class CbGeoMapShader;

public:
    GLGeoMapHandler (const char *name = "GLGeoMapHandler");
    ~GLGeoMapHandler ();

    bool set_lookup_table (const PointFloat2 *data, uint32_t width, uint32_t height);

    XCamReturn remap (const SmartPtr<VideoBuffer> &in_buf, SmartPtr<VideoBuffer> &out_buf);

    //derived from ImageHandler
    virtual XCamReturn terminate ();

protected:
    //derived from GLImageHandler
    virtual XCamReturn configure_resource (const SmartPtr<Parameters> &param);
    virtual XCamReturn start_work (const SmartPtr<Parameters> &param);

private:
    virtual bool init_factors ();

    virtual SmartPtr<GLGeoMapShader> create_geomap_shader ();
    virtual XCamReturn start_geomap_shader (const SmartPtr<ImageHandler::Parameters> &param);
    virtual void geomap_shader_done (
        const SmartPtr<Worker> &worker, const SmartPtr<Worker::Arguments> &args, const XCamReturn error);

    XCAM_DEAD_COPY (GLGeoMapHandler);

protected:
    SmartPtr<GLBuffer>              _lut_buf;
    SmartPtr<GLGeoMapShader>        _geomap_shader;
};

class GLDualConstGeoMapHandler
    : public GLGeoMapHandler
{
public:
    GLDualConstGeoMapHandler (const char *name = "GLDualConstGeoMapHandler");
    ~GLDualConstGeoMapHandler ();

    bool set_left_factors (float x, float y);
    void get_left_factors (float &x, float &y) {
        x = _left_factor_x;
        y = _left_factor_y;
    }
    bool set_right_factors (float x, float y);
    void get_right_factors (float &x, float &y) {
        x = _right_factor_x;
        y = _right_factor_y;
    }

private:
    virtual bool init_factors ();
    virtual XCamReturn start_geomap_shader (const SmartPtr<ImageHandler::Parameters> &param);

private:
    float        _left_factor_x;
    float        _left_factor_y;
    float        _right_factor_x;
    float        _right_factor_y;
};

extern SmartPtr<GLImageHandler> create_gl_geo_mapper ();

}
#endif // XCAM_GL_GEOMAP_HANDER_H
