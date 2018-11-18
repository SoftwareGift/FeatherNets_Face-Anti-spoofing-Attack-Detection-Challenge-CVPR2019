/*
 * render_osg_camera_manipulator.h -  supports 3D interactive manipulators
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

#ifndef XCAM_OSG_CAMERA_MANIPULATOR_H
#define XCAM_OSG_CAMERA_MANIPULATOR_H

#include <osgGA/StandardManipulator>

namespace XCam {

class RenderOsgCameraManipulator
    : public osgGA::StandardManipulator
{
public:

    explicit RenderOsgCameraManipulator ();

    virtual ~RenderOsgCameraManipulator ();

    virtual void setByMatrix (const osg::Matrixd &matrix)
    {
        (void)matrix;
    }

    virtual void setByInverseMatrix (const osg::Matrixd &matrix)
    {
        (void)matrix;
    }

    virtual osg::Matrixd getMatrix () const;

    virtual osg::Matrixd getInverseMatrix () const;

    virtual void home (double currentTime);

    virtual void setTransformation (const osg::Vec3d &eye, const osg::Quat &rotation)
    {
        (void)eye;
        (void)rotation;
    }

    virtual void setTransformation (const osg::Vec3d &eye, const osg::Vec3d &center, const osg::Vec3d &up)
    {
        (void)eye;
        (void)center;
        (void)up;
    }

    virtual void getTransformation (osg::Vec3d &eye, osg::Quat &rotation) const
    {
        (void)eye;
        (void)rotation;
    }

    virtual void getTransformation (osg::Vec3d &eye, osg::Vec3d &center, osg::Vec3d &up) const
    {
        (void)eye;
        (void)center;
        (void)up;
    }

    void setInitialValues (float angle, float length, float width, float height)
    {
        mAngle = angle;
        mLength = length;
        mWidth = width;
        mHeight = height;
        mMinHeight = height / 2.0f;

        mUp = osg::Vec3d(0.0f, 0.0f, 1.0f);
    }

private:
    RenderOsgCameraManipulator (RenderOsgCameraManipulator const &other);

    RenderOsgCameraManipulator &operator= (RenderOsgCameraManipulator const &other);

    virtual bool handleKeyDown (const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &us);

    virtual bool handleMouseWheel (const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &us);

    virtual bool performMovementLeftMouseButton (const double eventTimeDelta, const double dx, const double dy);

    void rotate (float deltaAngle);

    void modifyHeight (float delta);

    void getEyePosition (osg::Vec3d &eye) const;

    void getLookAtPosition (osg::Vec3d &center) const;

    float mAngle;
    float mLookAtOffset;
    float mMaxLookAtOffset;
    float mLength;
    float mWidth;
    float mHeight;
    float mMaxHeight;
    float mMinHeight;
    float mEyePosScale;
    osg::Vec3d mUp;
};

} // namespace XCam

#endif // XCAM_OSG_CAMERA_MANIPULATOR_H
