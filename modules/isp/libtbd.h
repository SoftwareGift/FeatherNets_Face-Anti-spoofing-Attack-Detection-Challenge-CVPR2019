/*
** Copyright 2012-2013 Intel Corporation
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
**     http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/

/*
 * \file libtbd.h
 * \brief Tagged Binary Data handling
 */

#ifndef __LIBTBD_H__
#define __LIBTBD_H__

#include <stddef.h>  /* defines size_t */
#include <stdint.h>  /* defines integer types with specified widths */
#include <stdio.h>   /* defines FILE */

#ifdef __cplusplus
extern "C" {
#endif

/*!
 *  Revision of TBD System, format 0xYYMMDDVV, where:
 * - YY: year,
 * - MM: month,
 * - DD: day,
 * - VV: version ('01','02' etc.)
 */
#define IA_TBD_VERSION 0x12032201

/*!
 *  Revision of TBD data set, format 0xYYMMDDVV, where:
 * - YY: year,
 * - MM: month,
 * - DD: day,
 * - VV: version ('01','02' etc.)
 */
#define IA_TBD_REVISION 0x13091001

/*!
* \brief Error codes for libtbd.
*/
typedef enum
{
    tbd_err_none     =       0 ,  /*!< No errors */
    tbd_err_general  = (1 << 1),  /*!< General error */
    tbd_err_nomemory = (1 << 2),  /*!< Out of memory */
    tbd_err_data     = (1 << 3),  /*!< Corrupted data */
    tbd_err_internal = (1 << 4),  /*!< Error in code */
    tbd_err_argument = (1 << 5)   /*!< Invalid argument for a function */
} tbd_error_t;

/*!
 * \brief Header structure for TBD container, followed by actual records.
 */
typedef struct
{
    uint32_t tag;          /*!< Tag identifier, also checks endianness */
    uint32_t size;         /*!< Container size including this header */
    uint32_t version;      /*!< Version of TBD system, format 0xYYMMDDVV */
    uint32_t revision;     /*!< Revision of TBD data set, format 0xYYMMDDVV */
    uint32_t config_bits;  /*!< Configuration flag bits set */
    uint32_t checksum;     /*!< Global checksum, header included */
} tbd_header_t;

/*!
 * \brief Tag identifiers used in TBD container header.
 */
#define CHTOU32(a,b,c,d) ((uint32_t)(a)|((uint32_t)(b)<<8)|((uint32_t)(c)<<16)|((uint32_t)(d)<<24))
typedef enum
{
    tbd_tag_cpff = CHTOU32('C', 'P', 'F', 'F'), /*!< CPF File */
    tbd_tag_aiqb = CHTOU32('A', 'I', 'Q', 'B'), /*!< AIQ configuration */
    tbd_tag_aiqd = CHTOU32('A', 'I', 'Q', 'D'), /*!< AIQ data */
    tbd_tag_halb = CHTOU32('H', 'A', 'L', 'B'), /*!< CameraHAL configuration */
    tbd_tag_drvb = CHTOU32('D', 'R', 'V', 'B') /*!< Sensor driver configuration */
} tbd_tag_t;

/*!
 * \brief Record structure. Data is located right after this header.
 */
typedef struct
{
    uint32_t size;        /*!< Size of record including header */
    uint8_t format_id;    /*!< tbd_format_t enumeration values used */
    uint8_t packing_key;  /*!< Packing method; 0 = no packing */
    uint16_t class_id;    /*!< tbd_class_t enumeration values used */
} tbd_record_header_t;

/*!
 * \brief Format ID enumeration describes the data format of the record.
 */
typedef enum
{
    tbd_format_any = 0,   /*!< Unspecified format */
    tbd_format_custom,    /*!< User specified format */
    tbd_format_container  /*!< Record is actually another TBD container */
} tbd_format_t;

/*!
 * \brief Class ID enumeration describes the data class of the record.
 */
typedef enum
{
    tbd_class_any = 0,  /*!< Unspecified record class */
    tbd_class_aiq,      /*!< Used for AIC and 3A records */
    tbd_class_drv,      /*!< Used for driver records */
    tbd_class_hal       /*!< Used for HAL records */
} tbd_class_t;

/*!
 * \brief Creates a new Tagged Binary Data container.
 * Creates a new, empty Tagged Binary Data container with the tag
 * that was given. Also updates the checksum and size accordingly.
 * Note that the buffer size must be large enough for the header
 * to fit in, the exact amount being 24 bytes (for tbd_header_t).
 * @param[in]    a_data_ptr        Pointer to modifiable container buffer
 * @param[in]    a_data_size       Size of the container buffer
 * @param[in]    a_tag             Tag the container shall have
 * @param[out]   a_new_size        Updated container size
 * @return                         Return code indicating possible errors
 */
tbd_error_t tbd_create(void *a_data_ptr,
                       size_t a_data_size,
                       tbd_tag_t a_tag,
                       size_t *a_new_size);

/*!
 * \brief Checks if Tagged Binary Data is valid. All tags are accepted.
 * Performs number of checks to given Tagged Binary Data container,
 * including the verification of the checksum. The function does not
 * care about the tag type of the container.
 * @param[in]    a_data_ptr        Pointer to container buffer
 * @param[in]    a_data_size       Size of the container buffer
 * @return                         Return code indicating possible errors
 */
tbd_error_t tbd_validate_anytag(void *a_data_ptr,
                                size_t a_data_size);

/*!
 * \brief Checks if Tagged Binary Data is valid, and tagged properly.
 * Performs number of checks to given Tagged Binary Data container,
 * including the verification of the checksum. Also, the data must have
 * been tagged properly. The tag is further used to check endianness,
 * and if it seems wrong, a specific debug message is printed out.
 * @param[in]    a_data_ptr        Pointer to container buffer
 * @param[in]    a_data_size       Size of the container buffer
 * @param[in]    a_tag             Tag the data must have
 * @return                         Return code indicating possible errors
 */
tbd_error_t tbd_validate(void *a_data_ptr,
                         size_t a_data_size,
                         tbd_tag_t a_tag);

/*!
 * \brief Finds a record of given kind from within the container.
 * Checks if a given kind of record exists in the Tagged Binary Data,
 * and if yes, tells the location of such record as well as its size.
 * If there are multiple records that match the query, the indicated
 * record is the first one.
 * @param[in]    a_data_ptr        Pointer to container buffer
 * @param[in]    a_record_class    Class the record must have
 * @param[in]    a_record_format   Format the record must have
 * @param[out]   a_record_data     Record data (or NULL if not found)
 * @param[out]   a_record_size     Record size (or 0 if not found)
 * @return                         Return code indicating possible errors
 */
tbd_error_t tbd_get_record(void *a_data_ptr,
                           tbd_class_t a_record_class,
                           tbd_format_t a_record_format,
                           void **a_record_data,
                           uint32_t *a_record_size);

/*!
 * \brief Updates the Tagged Binary Data with the given record inserted.
 * The given record is inserted into the Tagged Binary Data container
 * that must exist already. New records are always added to the end,
 * regardless if a record with the same class and format field already
 * exists in the data. Also updates the checksum and size accordingly.
 * Note that the buffer size must be large enough for the inserted
 * record to fit in, the exact amount being the size of original
 * Tagged Binary Data container plus the size of record data to be
 * inserted plus 8 bytes (for tbd_record_header_t).
 * @param[in]    a_data_ptr        Pointer to modifiable container buffer
 * @param[in]    a_data_size       Size of buffer (surplus included)
 * @param[in]    a_record_class    Class the record shall have
 * @param[in]    a_record_format   Format the record shall have
 * @param[in]    a_record_data     Record data
 * @param[in]    a_record_size     Record size
 * @param[out]   a_new_size        Updated container size
 * @return                         Return code indicating possible errors
 */
tbd_error_t tbd_insert_record(void *a_data_ptr,
                              size_t a_data_size,
                              tbd_class_t a_record_class,
                              tbd_format_t a_record_format,
                              void *a_record_data,
                              size_t a_record_size,
                              size_t *a_new_size);

/*!
 * \brief Updates the Tagged Binary Data with the given record removed.
 * The indicated record is removed from the Tagged Binary Data, after
 * which the checksum and size are updated accordingly. If there are
 * multiple records that match the class and format, only the first
 * instance is removed. If no record is found, nothing will be done.
 * Note that the resulting Tagged Binary Data container will
 * be smaller than the original, but it does not harm to store the
 * resulting container in its original length, either.
 * @param[in]    a_data_ptr        Pointer to modifiable container buffer
 * @param[in]    a_record_class    Class the record should have
 * @param[in]    a_record_format   Format the record should have
 * @param[out]   a_new_size        Updated container size
 * @return                         Return code indicating possible errors
 */
tbd_error_t tbd_remove_record(void *a_data_ptr,
                              tbd_class_t a_record_class,
                              tbd_format_t a_record_format,
                              size_t *a_new_size);

/*!
 * \brief Writes all possible information about the Tagged Binary Data.
 * Validates the Tagged Binary data container and generates a human
 * readable detailed report on the content, including information about
 * the records contained.
 * @param[in]    a_data_ptr        Pointer to container buffer
 * @param[in]    a_data_size       Size of the container buffer
 * @param[in]    a_outfile         Pointer to open file (may be stdout)
 * @return                         Return code indicating possible errors
 */
tbd_error_t tbd_infoprint(void *a_data_ptr,
                          size_t a_data_size,
                          FILE *a_outfile);

#ifdef __cplusplus
}
#endif

#endif /* __LIBTBD_H__ */
