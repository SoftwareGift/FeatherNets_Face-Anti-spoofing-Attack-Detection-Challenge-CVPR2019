/*
 *  Copyright (c) 2014 Intel Corporation
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

/*
 * \file xcam_cpf_reader.h
 * \brief  xcam CPF reader
*/

#ifndef _XCAM_CPF_READER_H
#define _XCAM_CPF_READER_H

#include <base/xcam_common.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>

XCAM_BEGIN_DECLARE


typedef int boolean;

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

typedef struct _XCamCpfBlob XCamCpfBlob;

/*! \brief CPF blob
 */
struct _XCamCpfBlob {
    uint8_t *data; /*!< pointer to buffer*/
    uint32_t size; /*!< buffer size*/
};

/*! \brief XCam CPF blob allocation.
 *  buffer is initialized to zero
 *
 * \return    pointer to XCam CPF Blob
 */
XCamCpfBlob * xcam_cpf_blob_new ();

/*! \brief XCam CPF blob release.
 *  release the blob structure as well as the buffer inside it.
 *
 * \param[in,out]    pointer to XCam CPF Blob
 */
void xcam_cpf_blob_free (XCamCpfBlob *blob);

/*! \brief XCam CPF blob release.
 *  release the blob structure as well as the buffer inside it. Called in xcam_3a_init().
 *
 * \param[in]     cpf_file   CPF file name
 * \param[out]    aiq_cpf    pointer to XCam CPF Blob which will hold AIQ records
 * \param[out]    hal_cpf    pointer to XCam CPF HAL which will hold HAL records
 */
boolean xcam_cpf_read (const char *cpf_file, XCamCpfBlob *aiq_cpf, XCamCpfBlob *hal_cpf);

XCAM_END_DECLARE

#endif //_XCAM_CPF_READER_H
