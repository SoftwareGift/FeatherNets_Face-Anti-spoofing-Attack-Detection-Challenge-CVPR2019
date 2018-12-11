/*
 * render_osg_viewer.cpp -  renders a single view on to a single scene
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

#include "render_osg_viewer.h"
#include "render_osg_model.h"
#include "render_osg_camera_manipulator.h"

#include <string>

namespace XCam {

RenderOsgViewer::RenderOsgViewer ()
    : Thread ("RenderOsgViewerThread")
    , _viewer (NULL)
    , _model_groups (NULL)
    , _initialized (false)
{
    _viewer = new osgViewer::Viewer ();

    if (!_initialized) {
        initialize ();
    }
}

RenderOsgViewer::~RenderOsgViewer ()
{
    _viewer->setDone (true);
}

XCamReturn
RenderOsgViewer::initialize ()
{
    XCamReturn result = XCAM_RETURN_NO_ERROR;

    osg::GraphicsContext::WindowingSystemInterface* wsi = osg::GraphicsContext::getWindowingSystemInterface();
    uint32_t win_width = 1920;
    uint32_t win_height = 1080;
    if (wsi) {
        wsi->getScreenResolution (osg::GraphicsContext::ScreenIdentifier (0), win_width, win_height);
    }

    _viewer->setThreadingModel (osgViewer::Viewer::SingleThreaded);

    _viewer->setLightingMode (osg::View::SKY_LIGHT);
    _viewer->getLight()->setAmbient (osg::Vec4f(0.f, 0.f, 0.f, 1.f));
    _viewer->getLight()->setDiffuse (osg::Vec4d(0.4f, 0.4f, 0.4f, 1.f));
    _viewer->getLight()->setSpecular (osg::Vec4d(0.5f, 0.5f, 0.5f, 1.f));

    osg::ref_ptr<osgViewer::StatsHandler> stats_handler = new osgViewer::StatsHandler ();
    _viewer->addEventHandler (stats_handler);

    _viewer->setUpViewInWindow (0, 0, win_width, win_height);

    osg::ref_ptr<RenderOsgCameraManipulator> vp_manipulator = new RenderOsgCameraManipulator ();
    vp_manipulator->setInitialValues (osg::PI, 6.0f, 4.0f, 2.6f);
    _viewer->setCameraManipulator (vp_manipulator);
    _viewer->getCamera ()->setClearMask (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    _viewer->getCamera ()->setComputeNearFarMode (osg::Camera::DO_NOT_COMPUTE_NEAR_FAR);
    _viewer->getCamera ()->setClearColor (osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    _viewer->getCamera ()->setViewport (0, 0, win_width, win_height);

    _initialized = true;

    return result;
}

void
RenderOsgViewer::start_render ()
{
    Thread::start ();
}

void
RenderOsgViewer::stop_render ()
{
    Thread::stop ();
}

bool
RenderOsgViewer::loop ()
{
    if (!_viewer->done ())
    {
        _viewer->frame ();
    }

    return true;
}

void
RenderOsgViewer::set_camera_manipulator (osg::ref_ptr<osgGA::StandardManipulator> &manipulator)
{
    _viewer->setCameraManipulator (manipulator);
}

void
RenderOsgViewer::add_model (SmartPtr<RenderOsgModel> &model)
{
    if (!model.ptr ()) {
        return;
    }

    if (!_model_groups.ptr ()) {
        _model_groups = model;
    } else {
        _model_groups->append_model (model);
    }
}

void
RenderOsgViewer::validate_model_groups ()
{
    if (!_model_groups.ptr ()) {
        return;
    }
    _viewer->setSceneData (_model_groups->get_model ());
}

} // namespace XCam

