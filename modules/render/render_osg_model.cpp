/*
 * render_osg_model.cpp -  represents renderable things by model
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

#include "render_osg_model.h"

#include <iostream>
#include <string>

#include <osg/MatrixTransform>
#include <osg/Texture2D>
#include <osgDB/ReadFile>

namespace XCam {

RenderOsgModel::RenderOsgModel (const char *name, uint32_t width, uint32_t height)
    : _name (NULL)
    , _model (NULL)
    , _geode (NULL)
    , _program (NULL)
    , _texture (NULL)
{
    XCAM_LOG_DEBUG ("RenderOsgModel width(%d), height(%d) ", width, height);
    XCAM_ASSERT (name);
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    _model = new osg::Group ();
    _geode = new osg::Geode ();

    _texture = create_texture (width, height);
    if (_texture.get ()) {
        add_texture (_texture);
    }
}

RenderOsgModel::RenderOsgModel (const char *name, bool from_file)
    : _name (NULL)
    , _model (NULL)
    , _geode (NULL)
    , _program (NULL)
    , _texture (NULL)
{
    XCAM_LOG_DEBUG ("RenderOsgModel model name (%s) ", name);
    XCAM_ASSERT (name);
    if (name)
        _name = strndup (name, XCAM_MAX_STR_SIZE);

    _model = new osg::Group ();
    _geode = new osg::Geode ();

    if (from_file && NULL != _geode.get ()) {
        osg::ref_ptr<osg::Node> node = create_model_from_file (name);
        _geode->addChild (node);
    }
}

RenderOsgModel::~RenderOsgModel ()
{
    if (_name)
        xcam_free (_name);
}

osg::Node*
RenderOsgModel::create_model_from_file (const char *name)
{
    XCAM_LOG_DEBUG ("Invailide node name %s", name);
    if (name == NULL) {
        return NULL;
    }

    osg::ref_ptr<osg::Node> node = osgDB::readNodeFile (name);
    if (NULL == node.get ()) {
        XCAM_LOG_ERROR ("Read node file FAILD!!! node name %s", name);
    }

    return node.release();
}

void
RenderOsgModel::append_model (SmartPtr<RenderOsgModel> &child_model)
{
    osg::ref_ptr<osg::Group> model = get_model ();

    if (NULL == model.get () || NULL == child_model.ptr ()) {
        XCAM_LOG_ERROR ("Append child model ERROR!! NULL model  !!");
        return;
    }

    model->addChild (child_model->get_model ());
}

void
RenderOsgModel::append_geode (SmartPtr<RenderOsgModel> &child_model)
{
    osg::ref_ptr<osg::Group> model = get_model ();

    if (NULL == model.get () || NULL == child_model.ptr ()) {
        XCAM_LOG_ERROR ("Append child geode ERROR!! NULL model  !!");
        return;
    }

    model->addChild (child_model->get_geode ());
}

XCamReturn
RenderOsgModel::setup_shader_program (
    const char *name,
    osg::Shader::Type type,
    const char *source_text)
{
    XCAM_LOG_DEBUG ("setup shader program name(%s), type(%d)", name, type);
    XCamReturn result = XCAM_RETURN_NO_ERROR;

    if (NULL == _program.get ()) {
        _program = new osg::Program ();
        _program->setName (name);
    }

    _program->addShader (new osg::Shader (type, source_text));

    _model->getOrCreateStateSet ()->setAttributeAndModes (_program, osg::StateAttribute::ON);
    _model->getOrCreateStateSet ()->setMode (GL_DEPTH_TEST, osg::StateAttribute::ON);

    return result;
}

XCamReturn
RenderOsgModel::setup_vertex_model (
    BowlModel::VertexMap &vertices,
    BowlModel::PointMap &points,
    BowlModel::IndexVector &indices,
    float a,
    float b,
    float c)
{
    XCamReturn result = XCAM_RETURN_NO_ERROR;

    osg::ref_ptr<osg::Group> model = get_model ();
    osg::ref_ptr<osg::Geode> geode = get_geode ();
    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry ();

    geode->addDrawable (geometry);
    geometry->setUseVertexBufferObjects (true);

    osg::ref_ptr<osg::Vec3Array> vertex_array = new osg::Vec3Array ();
    osg::ref_ptr<osg::Vec2Array> tex_coord_array = new osg::Vec2Array ();
    osg::ref_ptr<osg::DrawElementsUInt> index_array = new osg::DrawElementsUInt (GL_TRIANGLE_STRIP, 0);

    osg::ref_ptr<osg::Vec3Array> normal_array = new osg::Vec3Array ();
    osg::ref_ptr<osg::Vec4Array> color_array = new osg::Vec4Array ();

    normal_array->push_back (osg::Vec3 (0, -1, 0));
    geometry->setNormalArray (normal_array);
    geometry->setNormalBinding (osg::Geometry::BIND_OVERALL);

    color_array->push_back (osg::Vec4 (1.0, 0.0, 0.0, 1.0));
    geometry->setColorArray (color_array);
    geometry->setColorBinding (osg::Geometry::BIND_OVERALL);

    for (uint32_t idx = 0; idx < vertices.size (); idx++) {
        vertex_array->push_back (
            osg::Vec3f (vertices[idx].x * a,
                        vertices[idx].y * b,
                        vertices[idx].z * c));
    }

    for (uint32_t idx = 0; idx < points.size (); idx++) {
        tex_coord_array->push_back (osg::Vec2f (points[idx].x, points[idx].y));
    }

    for (uint32_t idx = 0; idx < indices.size (); idx++) {
        index_array->push_back (indices[idx]);
    }

    geometry->setVertexArray (vertex_array.get ());

    if (points.size () > 0) {
        geometry->setTexCoordArray (0, tex_coord_array.get ());
    }

    if (indices.size () > 0) {
        geometry->addPrimitiveSet (index_array.get ());
    } else {
        geometry->addPrimitiveSet (new osg::DrawArrays (GL_TRIANGLE_FAN, 0, 4));
    }

    model->addChild (geode);

    return result;
}

XCamReturn
RenderOsgModel::setup_model_matrix (
    float translation_x,
    float translation_y,
    float translation_z,
    float rotation_x,
    float rotation_y,
    float rotation_z,
    float rotation_degrees)
{
    XCamReturn result = XCAM_RETURN_NO_ERROR;

    osg::ref_ptr<osg::Group> model = get_model ();
    osg::ref_ptr<osg::Geode> geode = get_geode ();

    const osg::Vec3f axis(rotation_x, rotation_y, rotation_z);
    osg::ref_ptr<osg::MatrixTransform> mat = new osg::MatrixTransform ();
    mat->setMatrix (osg::Matrix::scale (osg::Vec3 (1.0, 1.0, 1.0)) *
                    osg::Matrix::rotate ((rotation_degrees / 180.f) * osg::PI_2, axis) *
                    osg::Matrix::translate (osg::Vec3 (translation_x, translation_y, translation_z)));

    mat->addChild (geode);
    model->addChild (mat);

    return result;
}

XCamReturn
RenderOsgModel::update_texture (SmartPtr<VideoBuffer> &buffer)
{
    XCamReturn result = XCAM_RETURN_NO_ERROR;

    XCAM_LOG_DEBUG ("RenderOsgModel::update_texture ");

    if (NULL == _texture.get ()) {
        return XCAM_RETURN_ERROR_PARAM;
    }

    {
        SmartLock locker (_mutex);

        VideoBufferInfo info = buffer->get_video_info ();
        uint32_t image_width = info.width;
        uint32_t image_height = info.height;

        osg::ref_ptr<osg::Image> image_y = new osg::Image ();
        osg::ref_ptr<osg::Image> image_uv = new osg::Image ();

        uint8_t* image_buffer = buffer->map ();
        if (NULL == image_buffer) {
            result = XCAM_RETURN_ERROR_MEM;
        } else {
            uint8_t* src_y = image_buffer;
            uint8_t* src_uv = image_buffer + image_width * image_height;

            image_y->setImage (image_width, image_height, 1,
                               GL_LUMINANCE, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                               src_y, osg::Image::NO_DELETE);

            image_uv->setImage (image_width / 2, image_height / 2, 1,
                                GL_LUMINANCE, GL_RG, GL_UNSIGNED_BYTE,
                                src_uv, osg::Image::NO_DELETE);

            _texture->_texture_y->setImage (image_y);
            _texture->_texture_uv->setImage (image_uv);
        }

        buffer->unmap ();
    }

    return result;
}

NV12Texture*
RenderOsgModel::create_texture (uint32_t width, uint32_t height)
{
    osg::ref_ptr<NV12Texture> nv12 = new NV12Texture (width, height);

    nv12->_texture_y = new osg::Texture2D ();
    nv12->_texture_y->setImage (new osg::Image ());

    nv12->_texture_y->setInternalFormat (GL_LUMINANCE);
    nv12->_texture_y->setResizeNonPowerOfTwoHint (false);
    nv12->_texture_y->setNumMipmapLevels (0);
    nv12->_texture_y->setFilter (osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    nv12->_texture_y->setFilter (osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    nv12->_texture_y->setWrap (osg::Texture::WRAP_S, osg::Texture::CLAMP);
    nv12->_texture_y->setWrap (osg::Texture::WRAP_T, osg::Texture::CLAMP);

    nv12->_texture_y->setTextureWidth (width);
    nv12->_texture_y->setTextureHeight (height);

    nv12->_texture_uv = new osg::Texture2D ();
    nv12->_texture_uv->setImage (new osg::Image ());

    nv12->_texture_uv->setInternalFormat (GL_RG);
    nv12->_texture_uv->setResizeNonPowerOfTwoHint (false);
    nv12->_texture_uv->setNumMipmapLevels (0);
    nv12->_texture_uv->setFilter (osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    nv12->_texture_uv->setFilter (osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    nv12->_texture_uv->setWrap (osg::Texture::WRAP_S, osg::Texture::CLAMP);
    nv12->_texture_uv->setWrap (osg::Texture::WRAP_T, osg::Texture::CLAMP);

    nv12->_texture_uv->setTextureWidth (width / 2);
    nv12->_texture_uv->setTextureHeight (height / 2);

    return nv12.release ();
}

XCamReturn
RenderOsgModel::add_texture (osg::ref_ptr<NV12Texture> &texture)
{
    XCamReturn result = XCAM_RETURN_NO_ERROR;

    osg::ref_ptr<osg::Group> model = get_model ();

    if (NULL == model.get () || NULL == _texture.get ()) {
        return XCAM_RETURN_ERROR_PARAM;
    }

    model->getOrCreateStateSet()->setTextureAttribute(0, texture->_texture_y.get ());
    texture->_uniform_y = new osg::Uniform (osg::Uniform::SAMPLER_2D, "textureY");
    texture->_uniform_y->set (0);
    model->getOrCreateStateSet ()->addUniform (texture->_uniform_y.get ());

    model->getOrCreateStateSet ()->setTextureAttribute (1, texture->_texture_uv.get ());
    texture->_uniform_uv = new osg::Uniform (osg::Uniform::SAMPLER_2D, "textureUV");
    texture->_uniform_uv->set (1);
    model->getOrCreateStateSet ()->addUniform (texture->_uniform_uv.get ());

    return result;
}

} // namespace XCam
