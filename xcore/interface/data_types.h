/*
 * data_types.h - data types in interface
 *
 *  Copyright (c) 2017 Intel Corporation
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
 * Author: Wind Yuan <feng.yuan@intel.com>
 * Author: Yinhang Liu <yinhangx.liu@intel.com>
 */

#ifndef XCAM_INTERFACE_DATA_TYPES_H
#define XCAM_INTERFACE_DATA_TYPES_H

#include <xcam_std.h>

namespace XCam {

enum SurroundMode {
    SphereView = 0,
    BowlView = 1
};

struct Rect {
    int32_t pos_x, pos_y;
    int32_t width, height;

    Rect () : pos_x (0), pos_y (0), width (0), height (0) {}
    Rect (int32_t x, int32_t y, int32_t w, int32_t h) : pos_x (x), pos_y (y), width (w), height (h) {}
};

struct ImageCropInfo {
    uint32_t left;
    uint32_t right;
    uint32_t top;
    uint32_t bottom;

    ImageCropInfo () : left (0), right (0), top (0), bottom (0) {}
};

struct FisheyeInfo {
    float    center_x;
    float    center_y;
    float    wide_angle;
    float    radius;
    float    rotate_angle; // clockwise

    FisheyeInfo ()
        : center_x (0.0f), center_y (0.0f), wide_angle (0.0f)
        , radius (0.0f), rotate_angle (0.0f)
    {}
    bool is_valid () const {
        return wide_angle >= 1.0f && radius >= 1.0f;
    }
};

#define XCAM_INTRINSIC_MAX_POLY_SIZE 16

// current intrinsic parameters definition from Scaramuzza's approach
struct IntrinsicParameter {
    float xc;
    float yc;
    float c;
    float d;
    float e;
    uint32_t poly_length;

    float poly_coeff[XCAM_INTRINSIC_MAX_POLY_SIZE];

    IntrinsicParameter ()
        : xc (0.0f), yc (0.0f), c(0.0f), d (0.0f), e (0.0f), poly_length (0)
    {
        xcam_mem_clear (poly_coeff);
    }
};

struct ExtrinsicParameter {
    float trans_x;
    float trans_y;
    float trans_z;

    // angle degree
    float roll;
    float pitch;
    float yaw;

    ExtrinsicParameter ()
        : trans_x (0.0f), trans_y (0.0f), trans_z (0.0f)
        , roll (0.0f), pitch (0.0f), yaw (0.0f)
    {}
};

template <typename T>
struct Point2DT {
    T x, y;
    Point2DT () : x (0), y(0) {}
    Point2DT (const T px, const T py) : x (px), y(py) {}
};

template <typename T>
struct Point3DT {
    T x, y, z;
    Point3DT () : x (0), y(0), z(0) {}
    Point3DT (const T px, const T py, const T pz) : x (px), y(py), z(pz) {}
};

typedef Point2DT<int32_t> PointInt2;
typedef Point2DT<float> PointFloat2;

typedef Point3DT<int32_t> PointInt3;
typedef Point3DT<float> PointFloat3;

/*
 * Ellipsoid model
 *  x^2 / a^2 + y^2 / b^2 + (z-center_z)^2 / c^2 = 1
 * ground : z = 0
 * x_axis : front direction
 * y_axis : left direction
 * z_axis : up direction
 * wall_height : bowl height inside of view
 * ground_length: left direction distance from ellipsoid bottom edge to nearest side of the car in the view
 */
struct BowlDataConfig {
    float a, b, c;
    float angle_start, angle_end; // angle degree

    // unit mm
    float center_z;
    float wall_height;
    float ground_length;

    BowlDataConfig ()
        : a (6060.0f), b (4388.0f), c (3003.4f)
        , angle_start (90.0f), angle_end (270.0f)
        , center_z (1500.0f)
        , wall_height (3000.0f)
        , ground_length (2801.0f)
    {
        XCAM_ASSERT (fabs(center_z) <= c);
        XCAM_ASSERT (a > 0.0f && b > 0.0f && c > 0.0f);
        XCAM_ASSERT (wall_height >= 0.0f && ground_length >= 0.0f);
        XCAM_ASSERT (ground_length <= b * sqrt(1.0f - center_z * center_z / (c * c)));
        XCAM_ASSERT (wall_height <= center_z + c);
    }
};

}

#endif //XCAM_INTERFACE_DATA_TYPES_H
