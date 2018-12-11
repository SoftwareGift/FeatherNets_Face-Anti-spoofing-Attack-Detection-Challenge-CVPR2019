/*
 * render_osg_viewer.h -  renders a single view on to a single scene
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

#ifndef XCAM_OSG_RENDER_VIEWER_H
#define XCAM_OSG_RENDER_VIEWER_H

#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osgGA/StandardManipulator>

#include <interface/data_types.h>
#include <interface/stitcher.h>
#include <xcam_mutex.h>
#include <xcam_thread.h>

namespace XCam {

class RenderOsgModel;

class RenderOsgViewer
    : public Thread
{
public:

    explicit RenderOsgViewer ();

    virtual ~RenderOsgViewer ();

    osgViewer::Viewer *get_viewer () {
        return _viewer;
    }

    void set_camera_manipulator (osg::ref_ptr<osgGA::StandardManipulator> &manipulator);

    void add_model (SmartPtr<RenderOsgModel> &model);
    void validate_model_groups ();

    void start_render ();
    void stop_render ();

protected:
    virtual bool loop ();

private:
    XCAM_DEAD_COPY (RenderOsgViewer);
    XCamReturn initialize ();

private:
    osg::ref_ptr<osgViewer::Viewer> _viewer;
    SmartPtr<RenderOsgModel> _model_groups;
    bool _initialized;
};

} // namespace XCam

#endif // XCAM_OSG_RENDER_VIEWER_H
