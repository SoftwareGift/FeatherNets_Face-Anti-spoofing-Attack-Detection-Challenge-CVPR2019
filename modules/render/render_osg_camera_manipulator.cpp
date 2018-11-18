/*
 * render_osg_camera_manipulator.cpp - supports 3D interactive manipulators
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

#include "render_osg_camera_manipulator.h"

namespace XCam {

RenderOsgCameraManipulator::RenderOsgCameraManipulator ()
    : osgGA::StandardManipulator::StandardManipulator ()
    , mAngle (osg::PI)
    , mLookAtOffset (0.0f)
    , mMaxLookAtOffset (osg::PI_4)
    , mLength (4.0f)
    , mWidth (3.0f)
    , mHeight (1.6f)
    , mMaxHeight (4.0f)
    , mMinHeight (0.6)
    , mEyePosScale (1.0f)
    , mUp (osg::Vec3d(0.0f, 0.0f, 1.0f))
{
    setAllowThrow (false);
    setAutoComputeHomePosition (false);
}

RenderOsgCameraManipulator::~RenderOsgCameraManipulator ()
{
}

osg::Matrixd
RenderOsgCameraManipulator::getInverseMatrix () const
{
    osg::Vec3d eyePos;
    getEyePosition (eyePos);
    osg::Vec3d lookAtPos;
    getLookAtPosition (lookAtPos);
    return osg::Matrixd::lookAt (eyePos, lookAtPos, mUp);
}

osg::Matrixd
RenderOsgCameraManipulator::getMatrix () const
{
    osg::Matrixd matrix = getInverseMatrix ();
    return osg::Matrixd::inverse (matrix);
}

void
RenderOsgCameraManipulator::home (double /*currentTime*/)
{
    mAngle = osg::PI;
    mLookAtOffset = 0.0f;
    mEyePosScale = 1.0f;
}

void
RenderOsgCameraManipulator::rotate (float deltaAngle)
{
    if (deltaAngle > 0.) {
        if (mLookAtOffset < mMaxLookAtOffset) {
            mLookAtOffset = std::min (mLookAtOffset + deltaAngle, mMaxLookAtOffset);
        } else {
            mAngle += deltaAngle;
        }
    } else {
        if (mLookAtOffset > -mMaxLookAtOffset) {
            mLookAtOffset = std::max (mLookAtOffset + deltaAngle, -mMaxLookAtOffset);
        } else {
            mAngle += deltaAngle;
        }
    }
    if (mAngle > 2 * osg::PI) {
        mAngle -= 2 * osg::PI;
    } else if (mAngle < 0.0f) {
        mAngle += 2 * osg::PI;
    }
}

void
RenderOsgCameraManipulator::modifyHeight (float delta)
{
    if (delta > 0.0) {
        mHeight = std::min (mHeight + delta, mMaxHeight);
    } else {
        mHeight = std::max (mHeight + delta, mMinHeight);
    }
}

void
RenderOsgCameraManipulator::getEyePosition (osg::Vec3d &eyePos) const
{
    float indentFactor = 1.0f - (0.1f * ((mHeight - mMinHeight) / (mMaxHeight - mMinHeight)));
    eyePos[0] = cos (mAngle) * mLength * indentFactor;
    eyePos[1] = sin (mAngle) * mWidth * indentFactor;
    eyePos[2] = mHeight;
    eyePos *= mEyePosScale;
}

void
RenderOsgCameraManipulator::getLookAtPosition (osg::Vec3d &lookAtPos) const
{
    float lookAtAngle = mAngle + mLookAtOffset;
    lookAtPos[0] = cos (lookAtAngle) * mLength * 0.5f;
    lookAtPos[1] = sin (lookAtAngle) * mWidth * 0.5f;
    lookAtPos[2] = mHeight * 0.25f;
}

bool
RenderOsgCameraManipulator::handleKeyDown (const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &us)
{
    (void)us;
    bool eventHandled = false;
    int key = ea.getKey ();
    if (key == osgGA::GUIEventAdapter::KEY_Space) {
        home (ea.getTime ());

        eventHandled = true;
    } else if (key == osgGA::GUIEventAdapter::KEY_Left) {
        rotate (-0.1);
        eventHandled = true;
    } else if (key == osgGA::GUIEventAdapter::KEY_Right) {
        rotate (0.1);
        eventHandled = true;
    }

    return eventHandled;
}

bool
RenderOsgCameraManipulator::handleMouseWheel (const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &us)
{
    (void)us;
    bool eventHandled = false;
    osgGA::GUIEventAdapter::ScrollingMotion sm = ea.getScrollingMotion();

    if (sm == osgGA::GUIEventAdapter::SCROLL_DOWN || sm == osgGA::GUIEventAdapter::SCROLL_RIGHT) {
        rotate (0.1);
        eventHandled = true;
    } else if (osgGA::GUIEventAdapter::SCROLL_UP || sm == osgGA::GUIEventAdapter::SCROLL_LEFT) {
        rotate (-0.1);
        eventHandled = true;
    }

    return eventHandled;
}

bool
RenderOsgCameraManipulator::performMovementLeftMouseButton (const double eventTimeDelta, const double dx, const double dy)
{
    (void)eventTimeDelta;

    rotate (-2.0 * dx);
    modifyHeight (-dy);
    return true;
}

} // namespace XCam
