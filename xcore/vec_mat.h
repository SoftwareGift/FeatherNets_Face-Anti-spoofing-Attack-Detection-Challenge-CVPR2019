/*
 * vec_mat.h - vector and matrix defination & calculation
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

#ifndef XCAM_VECTOR_MATRIX_H
#define XCAM_VECTOR_MATRIX_H

#include <xcam_std.h>
#include <cmath>


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
#define DEGREE_2_RADIANS(x) (((x) * PI) / 180.0)
#endif

#ifndef RADIANS_2_DEGREE
#define RADIANS_2_DEGREE(x) (((x) * 180.0) / PI)
#endif

#define XCAM_VECT2_OPERATOR_VECT2(op)                       \
    Vector2<T> operator op (const Vector2<T>& b) const {    \
        return Vector2<T>(x op b.x, y op b.y);              \
    }                                                       \
    Vector2<T> &operator op##= (const Vector2<T>& b) {      \
        x op##= b.x;  y op##= b.y; return *this;            \
    }

#define XCAM_VECT2_OPERATOR_SCALER(op)                      \
    Vector2<T> operator op (const T& b) const {             \
        return Vector2<T>(x op b, y op b);                  \
    }                                                       \
    Vector2<T> &operator op##= (const T& b) {               \
        x op##= b;  y op##= b; return *this;                \
    }

template<class T>
class Vector2
{
public:

    T x;
    T y;

    Vector2 () : x(0), y(0) {};
    Vector2 (T _x, T _y) : x(_x), y(_y) {};

    template <typename New>
    Vector2<New> convert_to () const {
        Vector2<New> ret((New)(this->x), (New)(this->y));
        return ret;
    }

    Vector2<T>& operator = (const Vector2<T>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    template <typename Other>
    Vector2<T>& operator = (const Vector2<Other>& rhs)
    {
        x = rhs.x;
        y = rhs.y;
        return *this;
    }

    Vector2<T> operator - () const {
        return Vector2<T>(-x, -y);
    }

    XCAM_VECT2_OPERATOR_VECT2 (+)
    XCAM_VECT2_OPERATOR_VECT2 (-)
    XCAM_VECT2_OPERATOR_VECT2 (*)
    XCAM_VECT2_OPERATOR_VECT2 ( / )
    XCAM_VECT2_OPERATOR_SCALER (+)
    XCAM_VECT2_OPERATOR_SCALER (-)
    XCAM_VECT2_OPERATOR_SCALER (*)
    XCAM_VECT2_OPERATOR_SCALER ( / )

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

template<class T, uint32_t N>
class VectorN
{
public:

    VectorN ();
    VectorN (T x);
    VectorN (T x, T y);
    VectorN (T x, T y, T z);
    VectorN (T x, T y, T z, T w);
    VectorN (VectorN<T, 3> vec3, T w);

    inline VectorN<T, N>& operator = (const VectorN<T, N>& rhs);
    inline VectorN<T, N> operator - () const;
    inline bool operator == (const VectorN<T, N>& rhs) const;

    inline T& operator [] (uint32_t index) {
        XCAM_ASSERT(index < N);
        return data[index];
    }
    inline const T& operator [] (uint32_t index) const {
        XCAM_ASSERT(index < N);
        return data[index];
    }

    inline VectorN<T, N> operator + (const T rhs) const;
    inline VectorN<T, N> operator - (const T rhs) const;
    inline VectorN<T, N> operator * (const T rhs) const;
    inline VectorN<T, N> operator / (const T rhs) const;
    inline VectorN<T, N> operator += (const T rhs);
    inline VectorN<T, N> operator -= (const T rhs);
    inline VectorN<T, N> operator *= (const T rhs);
    inline VectorN<T, N> operator /= (const T rhs);

    inline VectorN<T, N> operator + (const VectorN<T, N>& rhs) const;
    inline VectorN<T, N> operator - (const VectorN<T, N>& rhs) const;
    inline VectorN<T, N> operator * (const VectorN<T, N>& rhs) const;
    inline VectorN<T, N> operator / (const VectorN<T, N>& rhs) const;
    inline VectorN<T, N> operator += (const VectorN<T, N>& rhs);
    inline VectorN<T, N> operator -= (const VectorN<T, N>& rhs);
    inline VectorN<T, N> operator *= (const VectorN<T, N>& rhs);
    inline VectorN<T, N> operator /= (const VectorN<T, N>& rhs);

    template <typename NEW> inline
    VectorN<NEW, N> convert_to () const;

    inline void zeros ();
    inline void set (T x, T y);
    inline void set (T x, T y, T z);
    inline void set (T x, T y, T z, T w);
    inline T magnitude () const;
    inline float distance (const VectorN<T, N>& vec) const;
    inline T dot (const VectorN<T, N>& vec) const;
    inline VectorN<T, N> lerp (T weight, const VectorN<T, N>& vec) const;

private:
    T data[N];

};


template<class T, uint32_t N> inline
VectorN<T, N>::VectorN ()
{
    for (uint32_t i = 0; i < N; i++) {
        data[i] = 0;
    }
}

template<class T, uint32_t N> inline
VectorN<T, N>::VectorN (T x) {
    data[0] = x;
}

template<class T, uint32_t N> inline
VectorN<T, N>::VectorN (T x, T y) {
    if (N >= 2) {
        data[0] = x;
        data[1] = y;
    }
}

template<class T, uint32_t N> inline
VectorN<T, N>::VectorN (T x, T y, T z) {
    if (N >= 3) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
}

template<class T, uint32_t N> inline
VectorN<T, N>::VectorN (T x, T y, T z, T w) {
    if (N >= 4) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }
}

template<class T, uint32_t N> inline
VectorN<T, N>::VectorN (VectorN<T, 3> vec3, T w) {
    if (N >= 4) {
        data[0] = vec3.data[0];
        data[1] = vec3.data[1];
        data[2] = vec3.data[2];
        data[3] = w;
    }
}

template<class T, uint32_t N> inline
VectorN<T, N>& VectorN<T, N>::operator = (const VectorN<T, N>& rhs) {
    for (uint32_t i = 0; i < N; i++) {
        data[i] = rhs.data[i];
    }

    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator - () const {
    for (uint32_t i = 0; i < N; i++) {
        data[i] = -data[i];
    }

    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator + (const T rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] + rhs;
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator - (const T rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] - rhs;
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator * (const T rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] * rhs;
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator / (const T rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] / rhs;
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator += (const T rhs) {
    for (uint32_t i = 0; i < N; i++) {
        data[i] += rhs;
    }
    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator -= (const T rhs) {
    for (uint32_t i = 0; i < N; i++) {
        data[i] -= rhs;
    }
    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator *= (const T rhs) {
    for (uint32_t i = 0; i < N; i++) {
        data[i] *= rhs;
    }
    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator /= (const T rhs) {
    for (uint32_t i = 0; i < N; i++) {
        data[i] /= rhs;
    }
    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator + (const VectorN<T, N>& rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] + rhs.data[i];
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator - (const VectorN<T, N>& rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] - rhs.data[i];
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator * (const VectorN<T, N>& rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] * rhs.data[i];
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator / (const VectorN<T, N>& rhs) const {
    VectorN<T, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result.data[i] = data[i] / rhs.data[i];
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator += (const VectorN<T, N>& rhs) {

    for (uint32_t i = 0; i < N; i++) {
        data[i] += rhs.data[i];
    }
    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator -= (const VectorN<T, N>& rhs) {

    for (uint32_t i = 0; i < N; i++) {
        data[i] -= rhs.data[i];
    }
    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator *= (const VectorN<T, N>& rhs) {

    for (uint32_t i = 0; i < N; i++) {
        data[i] *= rhs.data[i];
    }
    return *this;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::operator /= (const VectorN<T, N>& rhs) {

    for (uint32_t i = 0; i < N; i++) {
        data[i] /= rhs.data[i];
    }
    return *this;
}

template<class T, uint32_t N> inline
bool VectorN<T, N>::operator == (const VectorN<T, N>& rhs) const {
    for (uint32_t i = 0; i < N; i++) {
        if (data[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

template <class T, uint32_t N>
template <typename NEW>
VectorN<NEW, N> VectorN<T, N>::convert_to () const {
    VectorN<NEW, N> result;

    for (uint32_t i = 0; i < N; i++) {
        result[i] = (NEW)(this->data[i]);
    }
    return result;
}

template <class T, uint32_t N> inline
void VectorN<T, N>::zeros () {
    for (uint32_t i = 0; i < N; i++) {
        data[i] = (T)(0);
    }
}

template<class T, uint32_t N> inline
void VectorN<T, N>::set (T x, T y) {
    if (N >= 2) {
        data[0] = x;
        data[1] = y;
    }
}

template<class T, uint32_t N> inline
void VectorN<T, N>::set (T x, T y, T z) {
    if (N >= 3) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    }
}

template<class T, uint32_t N> inline
void VectorN<T, N>::set (T x, T y, T z, T w) {
    if (N >= 4) {
        data[0] - x;
        data[1] = y;
        data[2] = z;
        data[3] = w;
    }
}

template<class T, uint32_t N> inline
T VectorN<T, N>::magnitude () const {
    T result = 0;

    for (uint32_t i = 0; i < N; i++) {
        result += (data[i] * data[i]);
    }
    return (T) sqrtf(result);
}

template<class T, uint32_t N> inline
float VectorN<T, N>::distance (const VectorN<T, N>& vec) const {
    T result = 0;

    for (uint32_t i = 0; i < N; i++) {
        result += (vec.data[i] - data[i]) * (vec.data[i] - data[i]);
    }
    return sqrtf(result);
}

template<class T, uint32_t N> inline
T VectorN<T, N>::dot (const VectorN<T, N>& vec) const {
    T result = 0;

    for (uint32_t i = 0; i < N; i++) {
        result += (vec.data[i] * data[i]);
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> VectorN<T, N>::lerp (T weight, const VectorN<T, N>& vec) const {
    return (*this) + (vec - (*this)) * weight;
}

// NxN matrix in row major order
template<class T, uint32_t N>
class MatrixN
{
public:
    MatrixN ();
    MatrixN (VectorN<T, 2> a, VectorN<T, 2> b);
    MatrixN (VectorN<T, 3> a, VectorN<T, 3> b, VectorN<T, 3> c);
    MatrixN (VectorN<T, 4> a, VectorN<T, 4> b, VectorN<T, 4> c, VectorN<T, 4> d);

    inline void zeros ();
    inline void eye ();

    inline T& at (uint32_t row, uint32_t col) {
        XCAM_ASSERT(row < N && col < N);
        return data[row * N + col];
    };
    inline const T& at (uint32_t row, uint32_t col) const {
        XCAM_ASSERT(row < N && col < N);
        return data[row * N + col];
    };

    inline T& operator () (uint32_t row, uint32_t col) {
        return at (row, col);
    };
    inline const T& operator () (uint32_t row, uint32_t col) const {
        return at (row, col);
    };

    inline MatrixN<T, N>& operator = (const MatrixN<T, N>& rhs);
    inline MatrixN<T, N> operator - () const;
    inline MatrixN<T, N> operator + (const MatrixN<T, N>& rhs) const;
    inline MatrixN<T, N> operator - (const MatrixN<T, N>& rhs) const;
    inline MatrixN<T, N> operator * (const T a) const;
    inline MatrixN<T, N> operator / (const T a) const;
    inline VectorN<T, N> operator * (const VectorN<T, N>& rhs) const;
    inline MatrixN<T, N> operator * (const MatrixN<T, N>& rhs) const;
    inline MatrixN<T, N> transpose ();
    inline MatrixN<T, N> inverse ();
    inline T trace ();

private:
    inline MatrixN<T, 2> inverse (const MatrixN<T, 2>& mat);
    inline MatrixN<T, 3> inverse (const MatrixN<T, 3>& mat);
    inline MatrixN<T, 4> inverse (const MatrixN<T, 4>& mat);

private:
    T data[N * N];

};

// NxN matrix in row major order
template<class T, uint32_t N>
MatrixN<T, N>::MatrixN () {
    eye ();
}

template<class T, uint32_t N>
MatrixN<T, N>::MatrixN (VectorN<T, 2> a, VectorN<T, 2> b) {
    if (N == 2) {
        data[0] = a[0];
        data[1] = a[1];
        data[2] = b[0];
        data[3] = b[1];
    } else {
        eye ();
    }
}

template<class T, uint32_t N>
MatrixN<T, N>::MatrixN (VectorN<T, 3> a, VectorN<T, 3> b, VectorN<T, 3> c) {
    if (N == 3) {
        data[0]  = a[0];
        data[1] = a[1];
        data[2] = a[2];
        data[3]  = b[0];
        data[4] = b[1];
        data[5] = b[2];
        data[6]  = c[0];
        data[7] = c[1];
        data[8] = c[2];
    } else {
        eye ();
    }
}

template<class T, uint32_t N>
MatrixN<T, N>::MatrixN (VectorN<T, 4> a, VectorN<T, 4> b, VectorN<T, 4> c, VectorN<T, 4> d) {
    if (N == 4) {
        data[0]  = a[0];
        data[1]  = a[1];
        data[2]  = a[2];
        data[3]  = a[3];
        data[4]  = b[0];
        data[5]  = b[1];
        data[6]  = b[2];
        data[7]  = b[3];
        data[8]  = c[0];
        data[9]  = c[1];
        data[10] = c[2];
        data[11] = c[3];
        data[12] = d[0];
        data[13] = d[1];
        data[14] = d[2];
        data[15] = d[3];
    } else {
        eye ();
    }
}

template<class T, uint32_t N> inline
void MatrixN<T, N>::zeros () {
    for (uint32_t i = 0; i < N * N; i++) {
        data[i] = 0;
    }
}

template<class T, uint32_t N> inline
void MatrixN<T, N>::eye () {
    zeros ();
    for (uint32_t i = 0; i < N; i++) {
        data[i * N + i] = 1;
    }
}

template<class T, uint32_t N> inline
MatrixN<T, N>& MatrixN<T, N>::operator = (const MatrixN<T, N>& rhs) {
    for (uint32_t i = 0; i < N * N; i++) {
        data[i] = rhs.data[i];
    }
    return *this;
}

template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::operator - () const {
    MatrixN<T, N> result;
    for (uint32_t i = 0; i < N * N; i++) {
        result.data[i] = -data[i];
    }
    return result;
}

template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::operator + (const MatrixN<T, N>& rhs) const {
    MatrixN<T, N> result;
    for (uint32_t i = 0; i < N * N; i++) {
        result.data[i] = data[i] + rhs.data[i];
    }
    return result;
}

template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::operator - (const MatrixN<T, N>& rhs) const {
    MatrixN<T, N> result;
    for (uint32_t i = 0; i < N * N; i++) {
        result.data[i] = data[i] - rhs.data[i];
    }
    return result;
}

template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::operator * (const T a) const {
    MatrixN<T, N> result;
    for (uint32_t i = 0; i < N * N; i++) {
        result.data[i] = data[i] * a;
    }
    return result;
}

template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::operator / (const T a) const {
    MatrixN<T, N> result;
    for (uint32_t i = 0; i < N * N; i++) {
        result.data[i] = data[i] / a;
    }
    return result;
}

template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::operator * (const MatrixN<T, N>& rhs) const {
    MatrixN<T, N> result;
    result.zeros ();

    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            T element = 0;
            for (uint32_t k = 0; k < N; k++) {
                element += at(i, k) * rhs(k, j);
            }
            result(i, j) = element;
        }
    }
    return result;
}

template<class T, uint32_t N> inline
VectorN<T, N> MatrixN<T, N>::operator * (const VectorN<T, N>& rhs) const {
    VectorN<T, N> result;
    for (uint32_t i = 0; i < N; i++) {  // row
        for (uint32_t j = 0; j < N; j++) {  // col
            result.data[i] = data[i * N + j] * rhs.data[j];
        }
    }
    return result;
}

template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::transpose () {
    MatrixN<T, N> result;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j <= N; j++) {
            result.data[i * N + j] = data[j * N + i];
        }
    }
    return result;
}

// if the matrix is non-invertible, return identity matrix
template<class T, uint32_t N> inline
MatrixN<T, N> MatrixN<T, N>::inverse () {
    MatrixN<T, N> result;

    result = inverse (*this);
    return result;
}

template<class T, uint32_t N> inline
T MatrixN<T, N>::trace () {
    T t = 0;
    for ( uint32_t i = 0; i < N; i++ ) {
        t += data(i, i);
    }
    return t;
}

template<class T, uint32_t N> inline
MatrixN<T, 2> MatrixN<T, N>::inverse (const MatrixN<T, 2>& mat)
{
    MatrixN<T, 2> result;

    T det = mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);

    if (det == (T)0) {
        return result;
    }

    result(0, 0) = mat(1, 1);
    result(0, 1) = -mat(0, 1);
    result(1, 0) = -mat(1, 0);
    result(1, 1) = mat(0, 0);

    return result * (1.0f / det);
}

template<class T, uint32_t N> inline
MatrixN<T, 3> MatrixN<T, N>::inverse (const MatrixN<T, 3>& mat)
{
    MatrixN<T, 3> result;

    T det = mat(0, 0) * mat(1, 1) * mat(2, 2) +
            mat(1, 0) * mat(2, 1) * mat(0, 2) +
            mat(2, 0) * mat(0, 1) * mat(1, 2) -
            mat(0, 0) * mat(2, 1) * mat(1, 2) -
            mat(1, 0) * mat(0, 1) * mat(2, 2) -
            mat(2, 0) * mat(1, 1) * mat(0, 2);

    if (det == (T)0) {
        return result;
    }

    result(0, 0) = mat(1, 1) * mat(2, 2) - mat(1, 2) * mat(2, 1);
    result(1, 0) = mat(1, 2) * mat(2, 0) - mat(1, 0) * mat(2, 2);
    result(2, 0) = mat(1, 0) * mat(2, 1) - mat(1, 1) * mat(2, 0);
    result(0, 1) = mat(0, 2) * mat(2, 1) - mat(0, 1) * mat(2, 2);
    result(1, 1) = mat(0, 0) * mat(2, 2) - mat(0, 2) * mat(2, 0);
    result(2, 1) = mat(0, 1) * mat(2, 0) - mat(0, 0) * mat(2, 1);
    result(0, 2) = mat(0, 1) * mat(1, 2) - mat(0, 2) * mat(1, 1);
    result(1, 2) = mat(0, 2) * mat(1, 0) - mat(0, 0) * mat(1, 2);
    result(2, 2) = mat(0, 0) * mat(1, 1) - mat(0, 1) * mat(1, 0);

    return result * (1.0f / det);
}

template<class T, uint32_t N> inline
MatrixN<T, 4> MatrixN<T, N>::inverse (const MatrixN<T, 4>& mat)
{
    MatrixN<T, 4> result;

    T det =  mat(0, 3) * mat(1, 2) * mat(2, 1) * mat(3, 1) -
             mat(0, 2) * mat(1, 3) * mat(2, 1) * mat(3, 1) -
             mat(0, 3) * mat(1, 1) * mat(2, 2) * mat(3, 1) +
             mat(0, 1) * mat(1, 3) * mat(2, 2) * mat(3, 1) +
             mat(0, 2) * mat(1, 1) * mat(2, 3) * mat(3, 1) -
             mat(0, 1) * mat(1, 2) * mat(2, 3) * mat(3, 1) -
             mat(0, 3) * mat(1, 2) * mat(2, 0) * mat(3, 1) +
             mat(0, 2) * mat(1, 3) * mat(2, 0) * mat(3, 1) +
             mat(0, 3) * mat(1, 0) * mat(2, 2) * mat(3, 1) -
             mat(0, 0) * mat(1, 3) * mat(2, 2) * mat(3, 1) -
             mat(0, 2) * mat(1, 0) * mat(2, 3) * mat(3, 1) +
             mat(0, 0) * mat(1, 2) * mat(2, 3) * mat(3, 1) +
             mat(0, 3) * mat(1, 1) * mat(2, 0) * mat(3, 2) -
             mat(0, 1) * mat(1, 3) * mat(2, 0) * mat(3, 2) -
             mat(0, 3) * mat(1, 0) * mat(2, 1) * mat(3, 2) +
             mat(0, 0) * mat(1, 3) * mat(2, 1) * mat(3, 2) +
             mat(0, 1) * mat(1, 0) * mat(2, 3) * mat(3, 2) -
             mat(0, 0) * mat(1, 1) * mat(2, 3) * mat(3, 2) -
             mat(0, 2) * mat(1, 1) * mat(2, 0) * mat(3, 3) +
             mat(0, 1) * mat(1, 2) * mat(2, 0) * mat(3, 3) +
             mat(0, 2) * mat(1, 0) * mat(2, 1) * mat(3, 3) -
             mat(0, 0) * mat(1, 2) * mat(2, 1) * mat(3, 3) -
             mat(0, 1) * mat(1, 0) * mat(2, 2) * mat(3, 3) +
             mat(0, 0) * mat(1, 1) * mat(2, 2) * mat(3, 3);

    if (det == (T)0) {
        return result;
    }

    result(0, 0) = mat(1, 2) * mat(2, 3) * mat(3, 1) -
                   mat(1, 3) * mat(2, 2) * mat(3, 1) +
                   mat(1, 3) * mat(2, 1) * mat(3, 2) -
                   mat(1, 1) * mat(2, 3) * mat(3, 2) -
                   mat(1, 2) * mat(2, 1) * mat(3, 3) +
                   mat(1, 1) * mat(2, 2) * mat(3, 3);

    result(0, 1) = mat(0, 3) * mat(2, 2) * mat(3, 1) -
                   mat(0, 2) * mat(2, 3) * mat(3, 1) -
                   mat(0, 3) * mat(2, 1) * mat(3, 2) +
                   mat(0, 1) * mat(2, 3) * mat(3, 2) +
                   mat(0, 2) * mat(2, 1) * mat(3, 3) -
                   mat(0, 1) * mat(2, 2) * mat(3, 3);

    result(0, 2) = mat(0, 2) * mat(1, 3) * mat(3, 1) -
                   mat(0, 3) * mat(1, 2) * mat(3, 1) +
                   mat(0, 3) * mat(1, 1) * mat(3, 2) -
                   mat(0, 1) * mat(1, 3) * mat(3, 2) -
                   mat(0, 2) * mat(1, 1) * mat(3, 3) +
                   mat(0, 1) * mat(1, 2) * mat(3, 3);

    result(0, 3) = mat(0, 3) * mat(1, 2) * mat(2, 1) -
                   mat(0, 2) * mat(1, 3) * mat(2, 1) -
                   mat(0, 3) * mat(1, 1) * mat(2, 2) +
                   mat(0, 1) * mat(1, 3) * mat(2, 2) +
                   mat(0, 2) * mat(1, 1) * mat(2, 3) -
                   mat(0, 1) * mat(1, 2) * mat(2, 3);

    result(1, 0) = mat(1, 3) * mat(2, 2) * mat(3, 0) -
                   mat(1, 2) * mat(2, 3) * mat(3, 0) -
                   mat(1, 3) * mat(2, 0) * mat(3, 2) +
                   mat(1, 0) * mat(2, 3) * mat(3, 2) +
                   mat(1, 2) * mat(2, 0) * mat(3, 3) -
                   mat(1, 0) * mat(2, 2) * mat(3, 3);

    result(1, 1) = mat(0, 2) * mat(2, 3) * mat(3, 0) -
                   mat(0, 3) * mat(2, 2) * mat(3, 0) +
                   mat(0, 3) * mat(2, 0) * mat(3, 2) -
                   mat(0, 0) * mat(2, 3) * mat(3, 2) -
                   mat(0, 2) * mat(2, 0) * mat(3, 3) +
                   mat(0, 0) * mat(2, 2) * mat(3, 3);

    result(1, 2) = mat(0, 3) * mat(1, 2) * mat(3, 0) -
                   mat(0, 2) * mat(1, 3) * mat(3, 0) -
                   mat(0, 3) * mat(1, 0) * mat(3, 2) +
                   mat(0, 0) * mat(1, 3) * mat(3, 2) +
                   mat(0, 2) * mat(1, 0) * mat(3, 3) -
                   mat(0, 0) * mat(1, 2) * mat(3, 3);

    result(1, 3) = mat(0, 2) * mat(1, 3) * mat(2, 0) -
                   mat(0, 3) * mat(1, 2) * mat(2, 0) +
                   mat(0, 3) * mat(1, 0) * mat(2, 2) -
                   mat(0, 0) * mat(1, 3) * mat(2, 2) -
                   mat(0, 2) * mat(1, 0) * mat(2, 3) +
                   mat(0, 0) * mat(1, 2) * mat(2, 3);

    result(2, 0) = mat(1, 1) * mat(2, 3) * mat(3, 0) -
                   mat(1, 3) * mat(2, 1) * mat(3, 0) +
                   mat(1, 3) * mat(2, 0) * mat(3, 1) -
                   mat(1, 0) * mat(2, 3) * mat(3, 1) -
                   mat(1, 1) * mat(2, 0) * mat(3, 3) +
                   mat(1, 0) * mat(2, 1) * mat(3, 3);

    result(2, 1) = mat(0, 3) * mat(2, 1) * mat(3, 0) -
                   mat(0, 1) * mat(2, 3) * mat(3, 0) -
                   mat(0, 3) * mat(2, 0) * mat(3, 1) +
                   mat(0, 0) * mat(2, 3) * mat(3, 1) +
                   mat(0, 1) * mat(2, 0) * mat(3, 3) -
                   mat(0, 0) * mat(2, 1) * mat(3, 3);

    result(2, 2) = mat(0, 1) * mat(1, 3) * mat(3, 0) -
                   mat(0, 3) * mat(1, 1) * mat(3, 0) +
                   mat(0, 3) * mat(1, 0) * mat(3, 1) -
                   mat(0, 0) * mat(1, 3) * mat(3, 1) -
                   mat(0, 1) * mat(1, 0) * mat(3, 3) +
                   mat(0, 0) * mat(1, 1) * mat(3, 3);

    result(2, 3) = mat(0, 3) * mat(1, 1) * mat(2, 0) -
                   mat(0, 1) * mat(1, 3) * mat(2, 0) -
                   mat(0, 3) * mat(1, 0) * mat(2, 1) +
                   mat(0, 0) * mat(1, 3) * mat(2, 1) +
                   mat(0, 1) * mat(1, 0) * mat(2, 3) -
                   mat(0, 0) * mat(1, 1) * mat(2, 3);

    result(3, 0) = mat(1, 2) * mat(2, 1) * mat(3, 0) -
                   mat(1, 1) * mat(2, 2) * mat(3, 0) -
                   mat(1, 2) * mat(2, 0) * mat(3, 1) +
                   mat(1, 0) * mat(2, 2) * mat(3, 1) +
                   mat(1, 1) * mat(2, 0) * mat(3, 2) -
                   mat(1, 0) * mat(2, 1) * mat(3, 2);

    result(3, 1) = mat(1, 1) * mat(2, 2) * mat(3, 0) -
                   mat(1, 2) * mat(2, 1) * mat(3, 0) +
                   mat(1, 2) * mat(2, 0) * mat(3, 1) -
                   mat(1, 0) * mat(2, 2) * mat(3, 1) -
                   mat(1, 1) * mat(2, 0) * mat(3, 2) +
                   mat(1, 0) * mat(2, 1) * mat(3, 2);

    result(3, 2) = mat(0, 2) * mat(1, 1) * mat(3, 0) -
                   mat(0, 1) * mat(1, 2) * mat(3, 0) -
                   mat(0, 2) * mat(1, 0) * mat(3, 1) +
                   mat(0, 0) * mat(1, 2) * mat(3, 1) +
                   mat(0, 1) * mat(1, 0) * mat(3, 2) -
                   mat(0, 0) * mat(1, 1) * mat(3, 2);

    result(3, 3) = mat(0, 1) * mat(1, 2) * mat(2, 0) -
                   mat(0, 2) * mat(1, 1) * mat(2, 0) +
                   mat(0, 2) * mat(1, 0) * mat(2, 1) -
                   mat(0, 0) * mat(1, 2) * mat(2, 1) -
                   mat(0, 1) * mat(1, 0) * mat(2, 2) +
                   mat(0, 0) * mat(1, 1) * mat(2, 2);

    return result * (1.0f / det);
}

typedef VectorN<double, 2> Vec2d;
typedef VectorN<double, 3> Vec3d;
typedef VectorN<double, 4> Vec4d;
typedef MatrixN<double, 2> Mat2d;
typedef MatrixN<double, 3> Mat3d;
typedef MatrixN<double, 4> Mat4d;

typedef VectorN<float, 2> Vec2f;
typedef VectorN<float, 3> Vec3f;
typedef VectorN<float, 4> Vec4f;
typedef MatrixN<float, 3> Mat3f;
typedef MatrixN<float, 4> Mat4f;

template<class T>
class Quaternion
{
public:

    Vec3d v;
    T w;

    Quaternion () : v(0, 0, 0), w(0) {};
    Quaternion (const Quaternion<T>& q) : v(q.v), w(q.w) {};

    Quaternion (const Vec3d& vec, T _w) : v(vec), w(_w) {};
    Quaternion (const Vec4d& vec)  : v(vec[0], vec[1], vec[2]), w(vec[3]) {};
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
        return Quaternion<T>(lhs.w * rhs.v[0] + lhs.v[0] * rhs.w + lhs.v[1] * rhs.v[2] - lhs.v[2] * rhs.v[1],
                             lhs.w * rhs.v[1] - lhs.v[0] * rhs.v[2] + lhs.v[1] * rhs.w + lhs.v[2] * rhs.v[0],
                             lhs.w * rhs.v[2] + lhs.v[0] * rhs.v[1] - lhs.v[1] * rhs.v[0] + lhs.v[2] * rhs.w,
                             lhs.w * rhs.w - lhs.v[0] * rhs.v[0] - lhs.v[1] * rhs.v[1] - lhs.v[2] * rhs.v[2]);
    }

    /*
                   --------
                  /    --
        |Qr| =  \/  Qr.Qr
    */
    inline T magnitude () const {
        return (T) sqrtf(w * w + v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
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
        T cos_theta = w * quat.w + v[0] * quat.v[0] + v[1] * quat.v[1] + v[2] * quat.v[2];
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
                ret.v[0] = v[0] * r0 + quat.v[0] * r1;
                ret.v[1] = v[1] * r0 + quat.v[1] * r1;
                ret.v[2] = v[2] * r0 + quat.v[2] * r1;
            }
        }
        return ret;
    }

    static Quaternion<T> create_quaternion (Vec3d axis, T angle_rad) {
        T theta_over_two = angle_rad / (T) 2.0;
        T sin_theta_over_two = std::sin(theta_over_two);
        T cos_theta_over_two = std::cos(theta_over_two);
        return Quaternion<T>(axis * sin_theta_over_two, cos_theta_over_two);
    }

    static Quaternion<T> create_quaternion (Vec3d euler) {
        return create_quaternion(Vec3d(1, 0, 0), euler[0]) *
               create_quaternion(Vec3d(0, 1, 0), euler[1]) *
               create_quaternion(Vec3d(0, 0, 1), euler[2]);
    }

    static Quaternion<T> create_quaternion (const Mat3d& mat) {
        Quaternion<T> q;

        T trace, s;
        T diag1 = mat(0, 0);
        T diag2 = mat(1, 1);
        T diag3 = mat(2, 2);

        trace = diag1 + diag2 + diag3;

        if (trace >= FLT_EPSILON)
        {
            s = 2.0 * (T) sqrt(trace + 1.0);
            q.w = 0.25 * s;
            q.v[0] = (mat(2, 1) - mat(1, 2)) / s;
            q.v[1] = (mat(0, 2) - mat(2, 0)) / s;
            q.v[2] = (mat(1, 0) - mat(0, 1)) / s;
        }
        else
        {
            char max_diag = (diag1 > diag2) ? ((diag1 > diag3) ? 1 : 3) : ((diag2 > diag3) ? 2 : 3);

            if (max_diag == 1)
            {
                s = 2.0 * (T) sqrt(1.0 + mat(0, 0) - mat(1, 1) - mat(2, 2));
                q.w = (mat(2, 1) - mat(1, 2)) / s;
                q.v[0] = 0.25 * s;
                q.v[1] = (mat(0, 1) + mat(1, 0)) / s;
                q.v[2] = (mat(0, 2) + mat(2, 0)) / s;
            }
            else if (max_diag == 2)
            {
                s = 2.0 * (T) sqrt(1.0 + mat(1, 1) - mat(0, 0) - mat(2, 2));
                q.w = (mat(0, 2) - mat(2, 0)) / s;
                q.v[0] = (mat(0, 1) + mat(1, 0)) / s;
                q.v[1] = 0.25 * s;
                q.v[2] = (mat(1, 2) + mat(2, 1)) / s;
            }
            else
            {
                s = 2.0 * (T) sqrt(1.0 + mat(2, 2) - mat(0, 0) - mat(1, 1));
                q.w = (mat(1, 0) - mat(0, 1)) / s;
                q.v[0] = (mat(0, 2) + mat(2, 0)) / s;
                q.v[1] = (mat(1, 2) + mat(2, 1)) / s;
                q.v[2] = 0.25 * s;
            }
        }

        return q;
    }

    inline Vec4d rotation_axis () {
        Vec4d rot_axis;

        T cos_theta_over_two = w;
        rot_axis[4] = (T) std::acos( cos_theta_over_two ) * 2.0f;

        T sin_theta_over_two = (T) sqrt( 1.0 - cos_theta_over_two * cos_theta_over_two );
        if ( fabs( sin_theta_over_two ) < 0.0005 ) sin_theta_over_two = 1;
        rot_axis[0] = v[0] / sin_theta_over_two;
        rot_axis[1] = v[1] / sin_theta_over_two;
        rot_axis[2] = v[2] / sin_theta_over_two;

        return rot_axis;
    }

    /*
        psi=atan2(2.*(Q(:,1).*Q(:,4)-Q(:,2).*Q(:,3)),(Q(:,4).^2-Q(:,1).^2-Q(:,2).^2+Q(:,3).^2));
        theta=asin(2.*(Q(:,1).*Q(:,3)+Q(:,2).*Q(:,4)));
        phi=atan2(2.*(Q(:,3).*Q(:,4)-Q(:,1).*Q(:,2)),(Q(:,4).^2+Q(:,1).^2-Q(:,2).^2-Q(:,3).^2));
    */
    inline Vec3d euler_angles () {
        Vec3d euler;

        // atan2(2*(qx*qw-qy*qz) , qw2-qx2-qy2+qz2)
        euler[0] = atan2(2 * (v[0] * w - v[1] * v[2]),
                         w * w - v[0] * v[0] - v[1] * v[1] + v[2] * v[2]);

        // asin(2*(qx*qz + qy*qw)
        euler[1] = asin(2 * (v[0] * v[2] + v[1] * w));

        // atan2(2*(qz*qw- qx*qy) , qw2 + qx2 - qy2 - qz2)
        euler[2] = atan2(2 * (v[2] * w - v[0] * v[1]),
                         w * w + v[0] * v[0] - v[1] * v[1] - v[2] * v[2]);

        return euler;
    }

    inline Mat3d rotation_matrix () {
        Mat3d mat;

        T xx = v[0] * v[0];
        T xy = v[0] * v[1];
        T xz = v[0] * v[2];
        T xw = v[0] * w;

        T yy = v[1] * v[1];
        T yz = v[1] * v[2];
        T yw = v[1] * w;

        T zz = v[2] * v[2];
        T zw = v[2] * w;

        mat(0, 0) = 1 - 2 * (yy + zz);
        mat(0, 1) = 2 * (xy - zw);
        mat(0, 2) = 2 * (xz + yw);
        mat(1, 0) = 2 * (xy + zw);
        mat(1, 1) = 1 - 2 * (xx + zz);
        mat(1, 2) = 2 * (yz - xw);
        mat(2, 0) = 2 * (xz - yw);
        mat(2, 1) = 2 * (yz + xw);
        mat(2, 2) = 1 - 2 * (xx + yy);

        return mat;
    }
};


typedef Quaternion<double> Quaternd;

}

#endif //XCAM_VECTOR_MATRIX_H
