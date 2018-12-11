/*
 * render_osg_model.h -  represents renderable things by object
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef XCAM_OSG_RENDER_MODEL_H
#define XCAM_OSG_RENDER_MODEL_H

#include <osg/Texture2D>
#include <osg/Group>

#include <interface/data_types.h>
#include <interface/stitcher.h>
#include <xcam_mutex.h>

namespace XCam {

class NV12Texture : public osg::Referenced
{
public:
    explicit NV12Texture (uint32_t width, uint32_t height)
    {
        _image_width = width;
        _image_height = height;
    }

public:
    uint32_t _image_width;
    uint32_t _image_height;

    osg::ref_ptr<osg::Texture2D> _texture_y;
    osg::ref_ptr<osg::Texture2D> _texture_uv;

    osg::ref_ptr<osg::Uniform> _uniform_y;
    osg::ref_ptr<osg::Uniform> _uniform_uv;
};

class RenderOsgModel {
public:

    explicit RenderOsgModel (const char *name, uint32_t width, uint32_t height);
    explicit RenderOsgModel (const char *name, bool from_file = true);

    virtual ~RenderOsgModel ();

    const char *get_name () const {
        return _name;
    }

    osg::Node* create_model_from_file (const char *name);

    osg::Group* get_model () const {
        return _model;
    }

    void append_model (SmartPtr<RenderOsgModel> &model);

    void append_geode (SmartPtr<RenderOsgModel> &model);


    osg::Geode* get_geode () const {
        return _geode;
    }

    XCamReturn setup_shader_program (
        const char *name,
        osg::Shader::Type type,
        const char *source_text);

    XCamReturn setup_vertex_model (
        BowlModel::VertexMap &vertices,
        BowlModel::PointMap &points,
        BowlModel::IndexVector &indices,
        float a = 1.0f,
        float b = 1.0f,
        float c = 1.0f);

    XCamReturn setup_model_matrix (
        float translation_x,
        float translation_y,
        float translation_z,
        float rotation_x,
        float rotation_y,
        float rotation_z,
        float rotation_degrees);

    XCamReturn update_texture (SmartPtr<VideoBuffer> &buffer);

private:

    XCAM_DEAD_COPY (RenderOsgModel);

    NV12Texture* create_texture (uint32_t width, uint32_t height);
    XCamReturn add_texture (osg::ref_ptr<NV12Texture> &texture);

private:

    char *_name;

    osg::ref_ptr<osg::Group> _model;
    osg::ref_ptr<osg::Geode> _geode;
    osg::ref_ptr<osg::Program> _program;
    osg::ref_ptr<NV12Texture> _texture;

    Mutex _mutex;
};

} // namespace XCam

#endif // XCAM_OSG_RENDER_MODEL_H
