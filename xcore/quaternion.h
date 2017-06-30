/*
 * quaternion.h - Quaternion defination & calculation
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
 * Author: Zong Wei <wei.zong@intel.com>
 */

#ifndef QUATERNOINS_H_DEF
#define QUATERNOINS_H_DEF

#include <cmath>
#include "xcam_utils.h"

namespace XCam {

#ifndef PI
#define PI 3.14159265358979323846
#endif

#ifndef FLT_EPSILON
#define FLT_EPSILON 1.19209290e-07F // float
#endif

#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-16 // double
#endif

#ifndef DEGREE_2_RADIANS
#define DEGREE_2_RADIANS(x) ((x * PI) / 180.0)
#endif

#ifndef RADIANS_2_DEGREE
#define RADIANS_2_DEGREE(x) ((x * 180) / PI)
#endif


template<class T>
class Vector2
{
public:

    T x;
    T y;

    Vector2 () : x(0), y(0) {};
    Vector2 (T _x, T _y) : x(_x), y(_y) {};

    Vector2<T>& operator = (const Vector2<T>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    Vector2<T> operator - () const {
        return Vector2<T>(-x, -y);
    }

    Vector2<T> operator + (const Vector2<T>& rhs) const {
        return Vector2<T>(x + rhs.x, y + rhs.y);
    }

    Vector2<T> operator - (const Vector2<T>& rhs) const {
        return Vector2<T>(x - rhs.x, y - rhs.y);
    }

    Vector2<T> operator * (const T a) const {
        return Vector2<T>(x * a, y * a);
    }

    Vector2<T> operator / (const T a) const {
        return Vector2<T>(x / a, y / a);
    }

    bool operator == (const Vector2<T>& rhs) const {
        return (x == rhs.x) && (y == rhs.y);
    }

    void reset () {
        this->x = (T) 0;
        this->y = (T) 0;
    }

    void set (T _x, T _y) {
        this->x = _x;
        this->y = _y;
    }

    T magnitude () const {
        return (T) sqrtf(x * x + y * y);
    }

    float distance (const Vector2<T>& vec) const {
        return sqrtf((vec.x - x) * (vec.x - x) + (vec.y - y) * (vec.y - y));
    }

    T dot (const Vector2<T>& vec) const {
        return (x * vec.x + y * vec.y);
    }

    inline Vector2<T> lerp (T weight, const Vector2<T>& vec) const {
        return (*this) + (vec - (*this)) * weight;
    }

};


template<class T>
class Vector3
{
public:

    T x;
    T y;
    T z;

    Vector3 () : x(0), y(0), z(0) {};
    Vector3 (T _x, T _y, T _z) : x(_x), y(_y), z(_z) {};

    Vector3<T>& operator = (const Vector3<T>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return *this;
    }

    inline Vector3<T> operator - () const {
        return Vector3<T>(-x, -y, -z);
    }

    inline Vector3<T> operator + (const Vector3<T>& rhs) const {
        return Vector3<T>(x + rhs.x, y + rhs.y, z + rhs.z);
    }

    inline Vector3<T> operator - (const Vector3<T>& rhs) const {
        return Vector3<T>(x - rhs.x, y - rhs.y, z - rhs.z);
    }

    inline Vector3<T> operator * (const T a) const {
        return Vector3<T>(x * a, y * a, z * a);
    }

    inline Vector3<T> operator / (const T a) const {
        return Vector3<T>(x / a, y / a, z / a);
    }


    inline bool operator == (const Vector3<T>& rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
    }

    inline void zeros () {
        this->x = (T) 0;
        this->y = (T) 0;
        this->z = (T) 0;
    }

    inline void set (T _x, T _y, T _z) {
        this->x = _x;
        this->y = _y;
        this->z = _z;
    }

    inline T magnitude () const {
        return (T) sqrtf(x * x + y * y + z * z);
    }

    inline float distance (const Vector3<T>& vec) const {
        return sqrtf((vec.x - x) * (vec.x - x) +
                     (vec.y - y) * (vec.y - y) +
                     (vec.z - z) * (vec.z - z));
    }

    inline T dot (const Vector3<T>& vec) const {
        return (x * vec.x + y * vec.y + z * vec.z);
    }

    inline Vector3<T> cross (const Vector3<T>& vec) const {
        return Vector3<T>(y * vec.z - z * vec.y, z * vec.x - x * vec.z, x * vec.y - y * vec.x);
    }

    inline Vector3<T> lerp (T weight, const Vector3<T>& vec) const {
        return (*this) + (vec - (*this)) * weight;
    }

};


template<class T>
class Vector4
{
public:

    T x;
    T y;
    T z;
    T w;

    Vector4 () : x(0), y(0), z(0), w(0) {};
    Vector4 (T _x, T _y, T _z, T _w) : x(_x), y(_y), z(_z), w(_w) {};
    Vector4 (Vector3<T> v, T _w) : x(v.x), y(v.y), z(v.z), w(_w) {};

    inline Vector4<T>& operator = (const Vector4<T>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        w = rhs.w;
        return *this;
    }

    inline Vector4<T> operator - () const {
        return Vector4<T>(-x, -y, -z, -w);
    }

    inline Vector4<T> operator + (const Vector4<T>& rhs) const {
        return Vector4<T>(x + rhs.x, y + rhs.y, z + rhs.z, w + rhs.w);
    }

    inline Vector4<T> operator - (const Vector4<T>& rhs) const {
        return Vector4<T>(x - rhs.x, y - rhs.y, z - rhs.z, w - rhs.w);
    }

    inline Vector4<T> operator * (const T a) const {
        return Vector4<T>(x * a, y * a, z * a, w * a);
    }

    inline Vector4<T> operator / (const T a) const {
        return Vector4<T>(x / a, y / a, z / a, w / a);
    }

    inline bool operator == (const Vector4<T>& rhs) const {
        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z) && (w == rhs.w);
    }

    inline void zeros () {
        this->x = (T) 0;
        this->y = (T) 0;
        this->z = (T) 0;
        this->w = (T) 0;
    }

    inline void set (T _x, T _y, T _z, T _w) {
        this->x = _x;
        this->y = _y;
        this->z = _z;
        this->w = _w;
    }

    inline T magnitude () const {
        return (T) sqrtf(x * x + y * y + z * z + w * w);
    }

    inline float distance (const Vector4<T>& vec) const {
        return sqrtf((vec.x - x) * (vec.x - x) +
                     (vec.y - y) * (vec.y - y) +
                     (vec.z - z) * (vec.z - z) +
                     (vec.w - w) * (vec.w - w));
    }

    inline T dot (const Vector4<T>& vec) const {
        return (x * vec.x + y * vec.y + z * vec.z + w * vec.w);
    }

    inline Vector4<T> lerp (T weight, const Vector4<T>& vec) const {
        return (*this) + (vec - (*this)) * weight;
    }

};


template<class T>
class Matrix3
{
public:

    // column vectors
    Vector3<T> v0;
    Vector3<T> v1;
    Vector3<T> v2;

    Matrix3 () : v0(1, 0, 0), v1(0, 1, 0), v2(0, 0, 1) {};
    Matrix3 (Vector3<T> a, Vector3<T> b, Vector3<T> c) : v0(a), v1(b), v2(c) {};

    inline void eye () {
        v0.set(1, 0, 0);
        v1.set(0, 1, 0);
        v2.set(0, 0, 1);
    }

    inline void zeros () {
        v0.zeros();
        v1.zeros();
        v2.zeros();
    }

    inline T& at (int row, int col) {
        XCAM_ASSERT(row >= 1 && row <= 3);
        XCAM_ASSERT(col >= 1 && col <= 3);

        if (col == 1 && row == 1) return v0.x;
        else if (col == 1 && row == 2) return v0.y;
        else if (col == 1 && row == 3) return v0.z;
        else if (col == 2 && row == 1) return v1.x;
        else if (col == 2 && row == 2) return v1.y;
        else if (col == 2 && row == 3) return v1.z;
        else if (col == 3 && row == 1) return v2.x;
        else if (col == 3 && row == 2) return v2.y;
        else if (col == 3 && row == 3) return v2.z;
        else return v0.x;
    }

    inline T& operator () (int row, int col) {
        XCAM_ASSERT(row >= 1 && row <= 3);
        XCAM_ASSERT(col >= 1 && col <= 3);

        if (col == 1 && row == 1) return v0.x;
        else if (col == 1 && row == 2) return v0.y;
        else if (col == 1 && row == 3) return v0.z;
        else if (col == 2 && row == 1) return v1.x;
        else if (col == 2 && row == 2) return v1.y;
        else if (col == 2 && row == 3) return v1.z;
        else if (col == 3 && row == 1) return v2.x;
        else if (col == 3 && row == 2) return v2.y;
        else if (col == 3 && row == 3) return v2.z;
        else return v0.x;
    }

    inline Matrix3<T>& operator = (const Matrix3<T>& rhs) {
        v0 = rhs.v0;
        v1 = rhs.v1;
        v2 = rhs.v2;
        return *this;
    }

    inline Matrix3<T> operator - () const {
        return Matrix3<T>(-v0, -v1, -v2);
    }

    inline Matrix3<T> operator + (const Matrix3<T>& rhs) const {
        return Matrix3<T>(v0 + rhs.v0, v1 + rhs.v1, v2 + rhs.v2);
    }

    inline Matrix3<T> operator * (const T a) const {
        return Matrix3<T>(v0 * a, v1 * a, v2 * a);
    }

    inline Matrix3<T> operator * (const Matrix3<T>& rhs) const {

        T m00 = Vector3<T>(v0.x, v1.x, v2.x).dot(rhs.v0);
        T m01 = Vector3<T>(v0.x, v1.x, v2.x).dot(rhs.v1);
        T m02 = Vector3<T>(v0.x, v1.x, v2.x).dot(rhs.v2);

        T m10 = Vector3<T>(v0.y, v1.y, v2.y).dot(rhs.v0);
        T m11 = Vector3<T>(v0.y, v1.y, v2.y).dot(rhs.v1);
        T m12 = Vector3<T>(v0.y, v1.y, v2.y).dot(rhs.v2);

        T m20 = Vector3<T>(v0.z, v1.z, v2.z).dot(rhs.v0);
        T m21 = Vector3<T>(v0.z, v1.z, v2.z).dot(rhs.v1);
        T m22 = Vector3<T>(v0.z, v1.z, v2.z).dot(rhs.v2);

        return Matrix3<T>(Vector3<T>(m00, m10, m20),
                          Vector3<T>(m01, m11, m21),
                          Vector3<T>(m02, m12, m22));
    }

    inline Vector3<T> operator * (const Vector3<T>& rhs) const {
        return Vector3<T>(v0.x * rhs.x + v1.x * rhs.y + v2.x * rhs.z,
                          v0.y * rhs.x + v1.y * rhs.y + v2.y * rhs.z,
                          v0.z * rhs.x + v1.z * rhs.y + v2.z * rhs.z);
    }

    inline Matrix3<T> transpose () {
        Matrix3<T> ret;
        for (int i = 1; i <= 3; i++)
        {
            for (int j = 1; j <= 3; j++)
            {
                ret.at(i, j) = at(j, i);
            }
        }
        return ret;
    }

    inline T det ()
    {
        return  at(1, 1) * at(2, 2) * at(3, 3) +
                at(2, 1) * at(3, 2) * at(1, 3) +
                at(3, 1) * at(1, 2) * at(2, 3) -
                at(1, 1) * at(3, 2) * at(2, 3) -
                at(2, 1) * at(1, 2) * at(3, 3) -
                at(3, 1) * at(2, 2) * at(1, 3);
    }

    inline Matrix3<T> inverse ()
    {
        Matrix3<T> ret;
        ret(1, 1) = at(2, 2) * at(3, 3) - at(2, 3) * at(3, 2);
        ret(2, 1) = at(2, 3) * at(3, 1) - at(2, 1) * at(3, 3);
        ret(3, 1) = at(2, 1) * at(3, 2) - at(2, 2) * at(3, 1);
        ret(1, 2) = at(1, 3) * at(3, 2) - at(1, 2) * at(3, 3);
        ret(2, 2) = at(1, 1) * at(3, 3) - at(1, 3) * at(3, 1);
        ret(3, 2) = at(1, 2) * at(3, 1) - at(1, 1) * at(3, 2);
        ret(1, 3) = at(1, 2) * at(2, 3) - at(1, 3) * at(2, 2);
        ret(2, 3) = at(1, 3) * at(2, 1) - at(1, 1) * at(2, 3);
        ret(3, 3) = at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1);
        return ret * (1.0f / det());
    }

};

template<class T>
class Matrix4
{
public:

    // column vectors
    Vector4<T> v0;
    Vector4<T> v1;
    Vector4<T> v2;
    Vector4<T> v3;

    Matrix4 () : v0(1, 0, 0, 0), v1(0, 1, 0, 0), v2(0, 0, 1, 0), v3(0, 0, 0, 1)  {};
    Matrix4 (Vector4<T> a, Vector4<T> b, Vector4<T> c, Vector4<T> d) : v0(a), v1(b), v2(c), v3(d) {};

    inline void eye () {
        v0.set(1, 0, 0, 0);
        v1.set(0, 1, 0, 0);
        v2.set(0, 0, 1, 0);
        v3.set(0, 0, 0, 1);
    }

    inline void zeros () {
        v0.zeros();
        v1.zeros();
        v2.zeros();
        v3.zeros();
    }

    inline T& at (int row, int col) {
        XCAM_ASSERT(row >= 1 && row <= 4);
        XCAM_ASSERT(col >= 1 && col <= 4);

        if (col == 1 && row == 1) return v0.x;
        else if (col == 1 && row == 2) return v0.y;
        else if (col == 1 && row == 3) return v0.z;
        else if (col == 1 && row == 4) return v0.w;
        else if (col == 2 && row == 1) return v1.x;
        else if (col == 2 && row == 2) return v1.y;
        else if (col == 2 && row == 3) return v1.z;
        else if (col == 2 && row == 4) return v1.w;
        else if (col == 3 && row == 1) return v2.x;
        else if (col == 3 && row == 2) return v2.y;
        else if (col == 3 && row == 3) return v2.z;
        else if (col == 3 && row == 4) return v2.w;
        else if (col == 4 && row == 1) return v3.x;
        else if (col == 4 && row == 2) return v3.y;
        else if (col == 4 && row == 3) return v3.z;
        else if (col == 4 && row == 4) return v3.w;
        else return v0.x;
    }

    inline T& operator () (int row, int col) {
        XCAM_ASSERT(row >= 1 && row <= 4);
        XCAM_ASSERT(col >= 1 && col <= 4);

        if (col == 1 && row == 1) return v0.x;
        else if (col == 1 && row == 2) return v0.y;
        else if (col == 1 && row == 3) return v0.z;
        else if (col == 1 && row == 4) return v0.w;
        else if (col == 2 && row == 1) return v1.x;
        else if (col == 2 && row == 2) return v1.y;
        else if (col == 2 && row == 3) return v1.z;
        else if (col == 2 && row == 4) return v1.w;
        else if (col == 3 && row == 1) return v2.x;
        else if (col == 3 && row == 2) return v2.y;
        else if (col == 3 && row == 3) return v2.z;
        else if (col == 3 && row == 4) return v2.w;
        else if (col == 4 && row == 1) return v3.x;
        else if (col == 4 && row == 2) return v3.y;
        else if (col == 4 && row == 3) return v3.z;
        else if (col == 4 && row == 4) return v3.w;
        else return v0.x;
    }

    inline Matrix4<T>& operator = (const Matrix4<T>& rhs) {
        v0 = rhs.v0;
        v1 = rhs.v1;
        v2 = rhs.v2;
        v3 = rhs.v3;
        return *this;
    }

    inline Matrix4<T> operator - () const {
        return Matrix4<T>(-v0, -v1, -v2, -v3);
    }

    inline Matrix4<T> operator + (const Matrix4<T>& rhs) const {
        return Matrix4<T>(v0 + rhs.v0, v1 + rhs.v1, v2 + rhs.v2, v3 + rhs.v3);
    }

    inline Matrix4<T> operator * (const T a) const {
        return Matrix4<T>(v0 * a, v1 * a, v2 * a, v3 * a);
    }

    inline Matrix4<T> operator * (const Matrix4<T>& rhs) const {
        T m00 = Vector4<T>(v0.x, v1.x, v2.x, v3.x).dot(rhs.v0);
        T m01 = Vector4<T>(v0.x, v1.x, v2.x, v3.x).dot(rhs.v1);
        T m02 = Vector4<T>(v0.x, v1.x, v2.x, v3.x).dot(rhs.v2);
        T m03 = Vector4<T>(v0.x, v1.x, v2.x, v3.x).dot(rhs.v3);

        T m10 = Vector4<T>(v0.y, v1.y, v2.y, v3.y).dot(rhs.v0);
        T m11 = Vector4<T>(v0.y, v1.y, v2.y, v3.y).dot(rhs.v1);
        T m12 = Vector4<T>(v0.y, v1.y, v2.y, v3.y).dot(rhs.v2);
        T m13 = Vector4<T>(v0.y, v1.y, v2.y, v3.y).dot(rhs.v3);

        T m20 = Vector4<T>(v0.z, v1.z, v2.z, v3.z).dot(rhs.v0);
        T m21 = Vector4<T>(v0.z, v1.z, v2.z, v3.z).dot(rhs.v1);
        T m22 = Vector4<T>(v0.z, v1.z, v2.z, v3.z).dot(rhs.v2);
        T m23 = Vector4<T>(v0.z, v1.z, v2.z, v3.z).dot(rhs.v3);

        T m30 = Vector4<T>(v0.w, v1.w, v2.w, v3.w).dot(rhs.v0);
        T m31 = Vector4<T>(v0.w, v1.w, v2.w, v3.w).dot(rhs.v1);
        T m32 = Vector4<T>(v0.w, v1.w, v2.w, v3.w).dot(rhs.v2);
        T m33 = Vector4<T>(v0.w, v1.w, v2.w, v3.w).dot(rhs.v3);

        return Matrix4<T>(Vector4<T>(m00, m10, m20, m30),
                          Vector4<T>(m01, m11, m21, m31),
                          Vector4<T>(m02, m12, m22, m32),
                          Vector4<T>(m03, m13, m23, m33));
    }

    inline Vector4<T> operator * (const Vector4<T>& rhs) const {
        return Vector4<T>(v0.x * rhs.x + v1.x * rhs.y + v2.x * rhs.z + v3.x * rhs.w,
                          v0.y * rhs.x + v1.y * rhs.y + v2.y * rhs.z + v3.y * rhs.w,
                          v0.z * rhs.x + v1.z * rhs.y + v2.z * rhs.z + v3.z * rhs.w,
                          v0.w * rhs.x + v1.w * rhs.y + v2.w * rhs.z + v3.w * rhs.w);
    }

    inline Matrix4<T> transpose () {
        Matrix4<T> ret;
        for (int i = 1; i <= 4; i++)
        {
            for (int j = 1; j <= 4; j++)
            {
                ret.at(i, j) = at(j, i);
            }
        }
        return ret;
    }

    inline T det()
    {
        return at(1, 4) * at(2, 3) * at(3, 2) * at(4, 1) -
               at(1, 3) * at(2, 4) * at(3, 2) * at(4, 1) -
               at(1, 4) * at(2, 2) * at(3, 3) * at(4, 1) +
               at(1, 2) * at(2, 4) * at(3, 3) * at(4, 1) +
               at(1, 3) * at(2, 2) * at(3, 4) * at(4, 1) -
               at(1, 2) * at(2, 3) * at(3, 4) * at(4, 1) -
               at(1, 4) * at(2, 3) * at(3, 1) * at(4, 2) +
               at(1, 3) * at(2, 4) * at(3, 1) * at(4, 2) +
               at(1, 4) * at(2, 1) * at(3, 3) * at(4, 2) -
               at(1, 1) * at(2, 4) * at(3, 3) * at(4, 2) -
               at(1, 3) * at(2, 1) * at(3, 4) * at(4, 2) +
               at(1, 1) * at(2, 3) * at(3, 4) * at(4, 2) +
               at(1, 4) * at(2, 2) * at(3, 1) * at(4, 3) -
               at(1, 2) * at(2, 4) * at(3, 1) * at(4, 3) -
               at(1, 4) * at(2, 1) * at(3, 2) * at(4, 3) +
               at(1, 1) * at(2, 4) * at(3, 2) * at(4, 3) +
               at(1, 2) * at(2, 1) * at(3, 4) * at(4, 3) -
               at(1, 1) * at(2, 2) * at(3, 4) * at(4, 3) -
               at(1, 3) * at(2, 2) * at(3, 1) * at(4, 4) +
               at(1, 2) * at(2, 3) * at(3, 1) * at(4, 4) +
               at(1, 3) * at(2, 1) * at(3, 2) * at(4, 4) -
               at(1, 1) * at(2, 3) * at(3, 2) * at(4, 4) -
               at(1, 2) * at(2, 1) * at(3, 3) * at(4, 4) +
               at(1, 1) * at(2, 2) * at(3, 3) * at(4, 4);
    }

    inline Matrix4<T> inverse()
    {
        Matrix4<T> ret;

        ret(1, 1) = at(2, 3) * at(3, 4) * at(4, 2) -
                    at(2, 4) * at(3, 3) * at(4, 2) +
                    at(2, 4) * at(3, 2) * at(4, 3) -
                    at(2, 2) * at(3, 4) * at(4, 3) -
                    at(2, 3) * at(3, 2) * at(4, 4) +
                    at(2, 2) * at(3, 3) * at(4, 4);

        ret(1, 2) = at(1, 4) * at(3, 3) * at(4, 2) -
                    at(1, 3) * at(3, 4) * at(4, 2) -
                    at(1, 4) * at(3, 2) * at(4, 3) +
                    at(1, 2) * at(3, 4) * at(4, 3) +
                    at(1, 3) * at(3, 2) * at(4, 4) -
                    at(1, 2) * at(3, 3) * at(4, 4);

        ret(1, 3) = at(1, 3) * at(2, 4) * at(4, 2) -
                    at(1, 4) * at(2, 3) * at(4, 2) +
                    at(1, 4) * at(2, 2) * at(4, 3) -
                    at(1, 2) * at(2, 4) * at(4, 3) -
                    at(1, 3) * at(2, 2) * at(4, 4) +
                    at(1, 2) * at(2, 3) * at(4, 4);

        ret(1, 4) = at(1, 4) * at(2, 3) * at(3, 2) -
                    at(1, 3) * at(2, 4) * at(3, 2) -
                    at(1, 4) * at(2, 2) * at(3, 3) +
                    at(1, 2) * at(2, 4) * at(3, 3) +
                    at(1, 3) * at(2, 2) * at(3, 4) -
                    at(1, 2) * at(2, 3) * at(3, 4);

        ret(2, 1) = at(2, 4) * at(3, 3) * at(4, 1) -
                    at(2, 3) * at(3, 4) * at(4, 1) -
                    at(2, 4) * at(3, 1) * at(4, 3) +
                    at(2, 1) * at(3, 4) * at(4, 3) +
                    at(2, 3) * at(3, 1) * at(4, 4) -
                    at(2, 1) * at(3, 3) * at(4, 4);

        ret(2, 2) = at(1, 3) * at(3, 4) * at(4, 1) -
                    at(1, 4) * at(3, 3) * at(4, 1) +
                    at(1, 4) * at(3, 1) * at(4, 3) -
                    at(1, 1) * at(3, 4) * at(4, 3) -
                    at(1, 3) * at(3, 1) * at(4, 4) +
                    at(1, 1) * at(3, 3) * at(4, 4);

        ret(2, 3) = at(1, 4) * at(2, 3) * at(4, 1) -
                    at(1, 3) * at(2, 4) * at(4, 1) -
                    at(1, 4) * at(2, 1) * at(4, 3) +
                    at(1, 1) * at(2, 4) * at(4, 3) +
                    at(1, 3) * at(2, 1) * at(4, 4) -
                    at(1, 1) * at(2, 3) * at(4, 4);

        ret(2, 4) = at(1, 3) * at(2, 4) * at(3, 1) -
                    at(1, 4) * at(2, 3) * at(3, 1) +
                    at(1, 4) * at(2, 1) * at(3, 3) -
                    at(1, 1) * at(2, 4) * at(3, 3) -
                    at(1, 3) * at(2, 1) * at(3, 4) +
                    at(1, 1) * at(2, 3) * at(3, 4);

        ret(3, 1) = at(2, 2) * at(3, 4) * at(4, 1) -
                    at(2, 4) * at(3, 2) * at(4, 1) +
                    at(2, 4) * at(3, 1) * at(4, 2) -
                    at(2, 1) * at(3, 4) * at(4, 2) -
                    at(2, 2) * at(3, 1) * at(4, 4) +
                    at(2, 1) * at(3, 2) * at(4, 4);

        ret(3, 2) = at(1, 4) * at(3, 2) * at(4, 1) -
                    at(1, 2) * at(3, 4) * at(4, 1) -
                    at(1, 4) * at(3, 1) * at(4, 2) +
                    at(1, 1) * at(3, 4) * at(4, 2) +
                    at(1, 2) * at(3, 1) * at(4, 4) -
                    at(1, 1) * at(3, 2) * at(4, 4);

        ret(3, 3) = at(1, 2) * at(2, 4) * at(4, 1) -
                    at(1, 4) * at(2, 2) * at(4, 1) +
                    at(1, 4) * at(2, 1) * at(4, 2) -
                    at(1, 1) * at(2, 4) * at(4, 2) -
                    at(1, 2) * at(2, 1) * at(4, 4) +
                    at(1, 1) * at(2, 2) * at(4, 4);

        ret(3, 4) = at(1, 4) * at(2, 2) * at(3, 1) -
                    at(1, 2) * at(2, 4) * at(3, 1) -
                    at(1, 4) * at(2, 1) * at(3, 2) +
                    at(1, 1) * at(2, 4) * at(3, 2) +
                    at(1, 2) * at(2, 1) * at(3, 4) -
                    at(1, 1) * at(2, 2) * at(3, 4);

        ret(4, 1) = at(2, 3) * at(3, 2) * at(4, 1) -
                    at(2, 2) * at(3, 3) * at(4, 1) -
                    at(2, 3) * at(3, 1) * at(4, 2) +
                    at(2, 1) * at(3, 3) * at(4, 2) +
                    at(2, 2) * at(3, 1) * at(4, 3) -
                    at(2, 1) * at(3, 2) * at(4, 3);

        ret(4, 2) = at(1, 2) * at(3, 3) * at(4, 1) -
                    at(1, 3) * at(3, 2) * at(4, 1) +
                    at(1, 3) * at(3, 1) * at(4, 2) -
                    at(1, 1) * at(3, 3) * at(4, 2) -
                    at(1, 2) * at(3, 1) * at(4, 3) +
                    at(1, 1) * at(3, 2) * at(4, 3);

        ret(4, 3) = at(1, 3) * at(2, 2) * at(4, 1) -
                    at(1, 2) * at(2, 3) * at(4, 1) -
                    at(1, 3) * at(2, 1) * at(4, 2) +
                    at(1, 1) * at(2, 3) * at(4, 2) +
                    at(1, 2) * at(2, 1) * at(4, 3) -
                    at(1, 1) * at(2, 2) * at(4, 3);

        ret(4, 4) = at(1, 2) * at(2, 3) * at(3, 1) -
                    at(1, 3) * at(2, 2) * at(3, 1) +
                    at(1, 3) * at(2, 1) * at(3, 2) -
                    at(1, 1) * at(2, 3) * at(3, 2) -
                    at(1, 2) * at(2, 1) * at(3, 3) +
                    at(1, 1) * at(2, 2) * at(3, 3);

        return ret * (1.0f / det());
    }


};


template<class T>
class Quaternion
{
public:

    Vector3<T> v;
    T w;

    Quaternion () : v(0, 0, 0), w(0) {};
    Quaternion (const Quaternion<T>& q) : v(q.v), w(q.w) {};

    Quaternion (const Vector3<T>& vec, T _w) : v(vec), w(_w) {};
    Quaternion (const Vector4<T>& vec) : v(vec.x, vec.y, vec.z), w(vec.w) {};
    Quaternion (T _x, T _y, T _z, T _w) : v(_x, _y, _z), w(_w) {};

    inline void reset () {
        v.zeros();
        w = (T) 0;
    }

    inline Quaternion<T>& operator = (const Quaternion<T>& rhs) {
        v = rhs.v;
        w = rhs.w;
        return *this;
    }

    inline Quaternion<T> operator + (const Quaternion<T>& rhs) const {
        const Quaternion<T>& lhs = *this;
        return Quaternion<T>(lhs.v + rhs.v, lhs.w + rhs.w);
    }

    inline Quaternion<T> operator - (const Quaternion<T>& rhs) const {
        const Quaternion<T>& lhs = *this;
        return Quaternion<T>(lhs.v - rhs.v, lhs.w - rhs.w);
    }

    inline Quaternion<T> operator * (T rhs) const {
        return Quaternion<T>(v * rhs, w * rhs);
    }

    inline Quaternion<T> operator * (const Quaternion<T>& rhs) const {
        const Quaternion<T>& lhs = *this;
        return Quaternion<T>(lhs.w * rhs.v.x + lhs.v.x * rhs.w + lhs.v.y * rhs.v.z - lhs.v.z * rhs.v.y,
                             lhs.w * rhs.v.y - lhs.v.x * rhs.v.z + lhs.v.y * rhs.w + lhs.v.z * rhs.v.x,
                             lhs.w * rhs.v.z + lhs.v.x * rhs.v.y - lhs.v.y * rhs.v.x + lhs.v.z * rhs.w,
                             lhs.w * rhs.w - lhs.v.x * rhs.v.x - lhs.v.y * rhs.v.y - lhs.v.z * rhs.v.z);
    }

    /*
                   --------
                  /    --
        |Qr| =  \/  Qr.Qr
    */
    inline T magnitude () const {
        return (T) sqrtf(w * w + v.x * v.x + v.y * v.y + v.z * v.z);
    }

    inline void normalize ()
    {
        T length = magnitude ();
        w = w / length;
        v = v / length;
    }

    inline Quaternion<T> conjugate (const Quaternion<T>& quat) const {
        return Quaternion<T>(-quat.v, quat.w);
    }

    inline Quaternion<T> inverse (const Quaternion<T>& quat) const {
        return conjugate(quat) * ( 1.0f / magnitude(quat));
    }

    inline Quaternion<T> lerp (T weight, const Quaternion<T>& quat) const {
        return Quaternion<T>(v.lerp(weight, quat.v), (1 - weight) * w + weight * quat.w);
    }

    inline Quaternion<T> slerp(T r, const Quaternion<T>& quat) const {
        Quaternion<T> ret;
        T cos_theta = w * quat.w + v.x * quat.v.x + v.y * quat.v.y + v.z * quat.v.z;
        T theta = (T) acos(cos_theta);
        if (fabs(theta) < FLT_EPSILON)
        {
            ret = *this;
        }
        else
        {
            T sin_theta = (T) sqrt(1.0 - cos_theta * cos_theta);
            if (fabs(sin_theta) < FLT_EPSILON)
            {
                ret.w = 0.5 * w + 0.5 * quat.w;
                ret.v = v.lerp(0.5, quat.v);
            }
            else
            {
                T r0 = (T) sin((1.0 - r) * theta) / sin_theta;
                T r1 = (T) sin(r * theta) / sin_theta;

                ret.w = w * r0 + quat.w * r1;
                ret.v.x = v.x * r0 + quat.v.x * r1;
                ret.v.y = v.y * r0 + quat.v.y * r1;
                ret.v.z = v.z * r0 + quat.v.z * r1;
            }
        }
        return ret;
    }

    static Quaternion<T> create_quaternion (Vector3<T> axis, T angle_rad) {
        T theta_over_two = angle_rad / (T) 2.0;
        T sin_theta_over_two = std::sin(theta_over_two);
        T cos_theta_over_two = std::cos(theta_over_two);
        return Quaternion<T>(axis * sin_theta_over_two, cos_theta_over_two);
    }

    static Quaternion<T> create_quaternion (Vector3<T> euler) {
        return create_quaternion(Vector3<T>(1, 0, 0), euler.x) *
               create_quaternion(Vector3<T>(0, 1, 0), euler.y) *
               create_quaternion(Vector3<T>(0, 0, 1), euler.z);
    }

    static Quaternion<T> create_quaternion (const Matrix3<T>& mat) {
        Quaternion<T> q;

        T trace, s;
        T diag1 = mat(1, 1);
        T diag2 = mat(2, 2);
        T diag3 = mat(3, 3);

        trace = diag1 + diag2 + diag3;

        if (trace >= FLT_EPSILON)
        {
            s = 2.0 * (T) sqrt(trace + 1.0);
            q.w = 0.25 * s;
            q.v.x = (mat(3, 2) - mat(2, 3)) / s;
            q.v.y = (mat(1, 3) - mat(3, 1)) / s;
            q.v.z = (mat(2, 1) - mat(1, 2)) / s;
        }
        else
        {
            char max_diag = (diag1 > diag2) ? ((diag1 > diag3) ? 1 : 3) : ((diag2 > diag3) ? 2 : 3);

            if (max_diag == 1)
            {
                s = 2.0 * (T) sqrt(1.0 + mat(1, 1) - mat(2, 2) - mat(3, 3));
                q.w = (mat(3, 2) - mat(2, 3)) / s;
                q.v.x = 0.25 * s;
                q.v.y = (mat(1, 2) + mat(2, 1)) / s;
                q.v.z = (mat(1, 3) + mat(3, 1)) / s;
            }
            else if (max_diag == 2)
            {
                s = 2.0 * (T) sqrt(1.0 + mat(2, 2) - mat(1, 1) - mat(3, 3));
                q.w = (mat(1, 3) - mat(3, 1)) / s;
                q.v.x = (mat(1, 2) + mat(2, 1)) / s;
                q.v.y = 0.25 * s;
                q.v.z = (mat(2, 3) + mat(3, 2)) / s;
            }
            else
            {
                s = 2.0 * (T) sqrt(1.0 + mat(3, 3) - mat(1, 1) - mat(2, 2));
                q.w = (mat(2, 1) - mat(1, 2)) / s;
                q.v.x = (mat(1, 3) + mat(3, 1)) / s;
                q.v.y = (mat(2, 3) + mat(3, 2)) / s;
                q.v.z = 0.25 * s;
            }
        }

        return q;
    }

    inline Vector4<T> rotation_axis () {
        Vector4<T> rot_axis;

        T cos_theta_over_two = w;
        rot_axis.w = (T) std::acos( cos_theta_over_two ) * 2.0f;

        T sin_theta_over_two = (T) sqrt( 1.0 - cos_theta_over_two * cos_theta_over_two );
        if ( fabs( sin_theta_over_two ) < 0.0005 ) sin_theta_over_two = 1;
        rot_axis.x = v.x / sin_theta_over_two;
        rot_axis.y = v.y / sin_theta_over_two;
        rot_axis.z = v.z / sin_theta_over_two;

        return rot_axis;
    }

    /*
        psi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
        theta=asin(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
        phi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
    */
    inline Vector3<T> euler_angles () {
        Vector3<T> euler;

        // atan2(2*(qx*qw-qy*qz) , qw2-qx2-qy2+qz2)
        euler.x = atan2(2 * (v.x * w - v.y * v.z),
                        w * w - v.x * v.x - v.y * v.y + v.z * v.z);

        // asin(2*(qx*qz + qy*qw)
        euler.y = asin(2 * (v.x * v.z + v.y * w));

        // atan2(2*(qz*qw- qx*qy) , qw2 + qx2 - qy2 - qz2)
        euler.z = atan2(2 * (v.z * w - v.x * v.y),
                        w * w + v.x * v.x - v.y * v.y - v.z * v.z);

        return euler;
    }

    inline Matrix3<T> rotation_matrix () {
        Matrix3<T> mat;

        T xx = v.x * v.x;
        T xy = v.x * v.y;
        T xz = v.x * v.z;
        T xw = v.x * w;

        T yy = v.y * v.y;
        T yz = v.y * v.z;
        T yw = v.y * w;

        T zz = v.z * v.z;
        T zw = v.z * w;

        mat(1, 1) = 1 - 2 * (yy + zz);
        mat(1, 2) = 2 * (xy - zw);
        mat(1, 3) = 2 * (xz + yw);
        mat(2, 1) = 2 * (xy + zw);
        mat(2, 2) = 1 - 2 * (xx + zz);
        mat(2, 3) = 2 * (yz - xw);
        mat(3, 1) = 2 * (xz - yw);
        mat(3, 2) = 2 * (yz + xw);
        mat(3, 3) = 1 - 2 * (xx + yy);

        return mat;
    }
};

typedef Vector2<double> Vec2d;
typedef Vector3<double> Vec3d;
typedef Vector4<double> Vec4d;
typedef Matrix3<double> Mat3d;
typedef Matrix4<double> Mat4d;
typedef Quaternion<double> Quaternd;

}

#endif

