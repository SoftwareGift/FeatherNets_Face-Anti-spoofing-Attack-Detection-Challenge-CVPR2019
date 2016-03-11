/*
 * x3a_result.cpp - 3A calculation result
 *
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

#include "x3a_result.h"

namespace XCam {

void
x3a_list_remove_result (X3aResultList &list, uint32_t type)
{
    for (X3aResultList::iterator i = list.begin (); i != list.end ();) {
        SmartPtr<X3aResult> &result = *i;
        XCAM_ASSERT (result.ptr ());
        if (result->get_type () == type) {
            list.erase (i++);
            continue;
        }
        ++i;
    }
}

};
