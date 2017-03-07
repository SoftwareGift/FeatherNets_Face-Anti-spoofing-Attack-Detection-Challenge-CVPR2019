/*
 * xcam_handle.h - image processing handles
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

#ifndef C_XCAM_HANDLE_H
#define C_XCAM_HANDLE_H

#include <base/xcam_defs.h>
#include <base/xcam_common.h>
#include <base/xcam_buffer.h>

XCAM_BEGIN_DECLARE

typedef struct _XCamHandle XCamHandle;

/*! \brief    create xcam handle to process buffer
 *
 * \params[in]    name, filter name
 * \return        XCamHandle    create correct hanle, else return NULL.
 */
XCamHandle *xcam_create_handle (const char *name);

/*! \brief    destroy xcam handle
 *
 * \params[in]    handle        handle need to destory
 */
void xcam_destroy_handle (XCamHandle *handle);

/*! \brief    xcam handle get usage how to set parameters
 *
 * \params[in]        handle       xcam handle
 * \params[out]       usage_buf    buffer to store usage
 * \params[in,out]    usage_len    buffer length
 * \return            XCamReturn   XCAM_RETURN_NO_ERROR on sucess; others on errors.
 */
XCamReturn xcam_handle_get_usage (XCamHandle *handle, char *usage_buf, int *usage_len);

/*! \brief set handle parameters before init
 *
 * \params[in]    handle       xcam handle
 * \params[in]    field0, value0, field1, value1, ..., fieldN, valueN    field and value in pairs
 * \return        XCamReturn   XCAM_RETURN_NO_ERROR on sucess; others on errors.
 */
XCamReturn xcam_handle_set_parameters (
    XCamHandle *handle, const char *field, ...);

/*! \brief    xcam handle initialize
 *
 * \params[in]        handle       xcam handle
 * \return            XCamReturn   XCAM_RETURN_NO_ERROR on sucess; others on errors.
 */
XCamReturn xcam_handle_init (XCamHandle *handle);

/*! \brief    xcam handle uninitialize
 *
 * \params[in]        handle       xcam handle
 * \return            XCamReturn   XCAM_RETURN_NO_ERROR on sucess; others on errors.
 */
XCamReturn xcam_handle_uinit (XCamHandle *handle);

// buf_out was allocated outside or inside ??
/*! \brief    xcam handle process buffer
 *
 * \params[in]        handle       xcam handle
 * \params[in]        buf_in       input buffer
 * \params[in,out]    buf_out      output buffer, can be allocated outside or inside,
 *                                 if set param "alloc-out-buf" "true", allocate outside; else inside.
 * \return            XCamReturn   XCAM_RETURN_NO_ERROR on sucess; others on errors.
 */
XCamReturn xcam_handle_execute (XCamHandle *handle, XCamVideoBuffer *buf_in, XCamVideoBuffer **buf_out);

XCAM_END_DECLARE

#endif //C_XCAM_HANDLE_H
