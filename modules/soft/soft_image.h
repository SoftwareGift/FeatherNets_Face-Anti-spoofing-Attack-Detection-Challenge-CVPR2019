/*
 * soft_image.h - soft image class
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
 */

#ifndef XCAM_SOFT_IMAGE_H
#define XCAM_SOFT_IMAGE_H

#include "xcam_utils.h"
#include "video_buffer.h"
#include "vec_mat.h"

namespace XCam {

typedef uint8_t Uchar;
typedef int8_t Char;
typedef Vector2<uint8_t> Uchar2;
typedef Vector2<int8_t> Char2;
typedef Vector2<float> Float2;


enum BorderType {
    BorderTypeNearest,
    BorderTypeConst,
    BorderTypeRewind,
};

template <typename T>
class SoftImage
{
private:
    uint8_t    *_buf_ptr;
    uint32_t    _width;
    uint32_t    _height;
    uint32_t    _pitch;

    SmartPtr<VideoBuffer> _bind;

public:
    explicit SoftImage (
        const SmartPtr<VideoBuffer> &buf,
        const uint32_t width,
        const uint32_t height,
        const uint32_t pictch,
        const uint32_t offset = 0)
        : _buf_ptr (NULL)
        , _width (width)
        , _height (height)
        , _pitch (pictch)
        , _bind (buf)
    {
        XCAM_ASSERT (buf.ptr ());
        XCAM_ASSERT (buf->map ());
        _buf_ptr = buf->map () + offset;
    }

    uint32_t get_width () const {
        return _width;
    }
    uint32_t get_height () const {
        return _height;
    }
    const SmartPtr<VideoBuffer> &get_buf () const {
        return _bind;
    }

    inline T read_data_no_check (int32_t x, int32_t y) {
        const T *t_ptr = (const T *)(_buf_ptr + y * _pitch);
        return t_ptr[x];
    }

    inline T read_data (int32_t x, int32_t y) {
        border_check (x, y);
        return read_data_no_check (x, y);
    }

    template<uint32_t N>
    inline void read_array_no_check (int32_t x, int32_t y, T *array) {
        XCAM_ASSERT (N <= 8);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch));
        memcpy (array, t_ptr + x, sizeof (T) * N);
    }

    template<uint32_t N>
    inline void read_array (int32_t x, int32_t y, T *array) {
        XCAM_ASSERT (N <= 8);
        border_check_y (y);
        if (x + N < _width) {
            read_array_no_check<N> (x, y, array);
        } else {
            const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch));
            for (uint32_t i = 0; i < N; ++i, ++x) {
                border_check_x (x);
                array[i] = t_ptr[x];
            }
        }
    }

    template<typename O, uint32_t N>
    inline void read_array (int32_t x, int32_t y, O *array) {
        XCAM_ASSERT (N <= 8);
        border_check_y (y);
        const T *t_ptr = ((const T *)(_buf_ptr + y * _pitch));
        for (uint32_t i = 0; i < N; ++i, ++x) {
            border_check_x (x);
            array[i] = t_ptr[x];
        }
    }

    inline void write_data (int32_t x, int32_t y, const T &v) {
        if (x < 0 || x >= (int32_t)_width)
            return;
        if (y < 0 || y >= (int32_t)_height)
            return;
        write_data_no_check (x, y, v);
    }

    inline void write_data_no_check (int32_t x, int32_t y, const T &v) {
        T *t_ptr = (T *)(_buf_ptr + y * _pitch);
        t_ptr[x] = v;
    }

    template<uint32_t N>
    inline void write_array_no_check (int32_t x, int32_t y, const T *array) {
        T *t_ptr = (T *)(_buf_ptr + y * _pitch);
        memcpy (t_ptr + x, array, sizeof (T) * N);
    }

    template<uint32_t N>
    inline void write_array (int32_t x, int32_t y, const T *array) {
        if (y < 0 || y >= (int32_t)_height)
            return;

        if (x >= 0 && x + N <= _width) {
            write_array_no_check<N> (x, y, array);
        } else {
            T *t_ptr = ((T *)(_buf_ptr + y * _pitch));
            for (uint32_t i = 0; i < N; ++i, ++x) {
                if (x < 0 || x >= (int32_t)_width) continue;
                t_ptr[x] = array[i];
            }
        }
    }

private:
    inline void border_check_x (int32_t &x) {
        if (x < 0) x = 0;
        else if (x >= (int32_t)_width) x = (int32_t)(_width - 1);
    }

    inline void border_check_y (int32_t &y) {
        if (y < 0) y = 0;
        else if (y >= (int32_t)_height) y = (int32_t)(_height - 1);
    }

    inline void border_check (int32_t &x, int32_t &y) {
        border_check_x (x);
        border_check_y (y);
    }
};

template <typename T>
inline Uchar convert_to_uchar (const T& v) {
    if (v < 0.0f) return 0;
    else if (v > 255.0f) return 255;
    return (Uchar)(v + 0.5f);
}

}
#endif //XCAM_SOFT_IMAGE_H