/*
 * cl_argument.h - CL kernel Argument
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

#ifndef XCAM_CL_KERNEL_ARGUMENT_H
#define XCAM_CL_KERNEL_ARGUMENT_H

#include <xcam_std.h>
#include <ocl/cl_memory.h>

namespace XCam {

#define XCAM_DEFAULT_IMAGE_DIM 2
#define XCAM_CL_KERNEL_MAX_WORK_DIM 3

struct CLWorkSize
{
    uint32_t dim;
    size_t global[XCAM_CL_KERNEL_MAX_WORK_DIM];
    size_t local[XCAM_CL_KERNEL_MAX_WORK_DIM];
    CLWorkSize();
};

class CLArgument
{
public:
    virtual ~CLArgument ();
    void get_value (void *&adress, uint32_t &size);

protected:
    CLArgument (uint32_t size);

private:
    XCAM_DEAD_COPY (CLArgument);

protected:
    void     *_arg_adress;
    uint32_t  _arg_size;
};

typedef std::list<SmartPtr<CLArgument> > CLArgList;


template<typename DataType>
class CLArgumentT
    : public CLArgument
{
public:

    CLArgumentT (const DataType &value)
        : CLArgument (sizeof (DataType))
        , _value (value)
    {
        _arg_adress = (void *) &_value;
    }
    ~CLArgumentT () {}

private:
    DataType     _value;
};

template<typename DataType, int count>
class CLArgumentTArray
    : public CLArgument
{
public:

    CLArgumentTArray (const DataType *value)
        : CLArgument (sizeof (DataType) * count)
    {
        memcpy (&_value[0], value, sizeof (DataType) * count);
        _arg_adress = (void *) &_value;
    }
    ~CLArgumentTArray () {}

private:
    DataType     _value[count];
};

class CLMemArgument
    : public CLArgument
{
public:

    CLMemArgument (const SmartPtr<CLMemory> &mem)
        : CLArgument (sizeof (cl_mem))
        , _mem (mem)
    {
        XCAM_ASSERT (mem.ptr ());
        _arg_adress = &mem->get_mem_id ();
    }
    ~CLMemArgument () {}

private:
    SmartPtr<CLMemory>  _mem;
};


}

#endif //XCAM_CL_KERNEL_ARGUMENT_H
