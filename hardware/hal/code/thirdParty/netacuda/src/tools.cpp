/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <tools.h>
#include <cstring>

void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorName(err));
}

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStream(int flags, int priority)
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithPriority(&stream, flags, priority));
    return std::unique_ptr<CUstream_st, StreamDeleter>{stream};
}

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStreamNew()
{
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    return std::unique_ptr<CUstream_st, StreamDeleter>{stream};
}

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEvent(int flags)
{
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreateWithFlags(&event, flags));
    return std::unique_ptr<CUevent_st, EventDeleter>{event};
}

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEventNew(void)
{
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));
    return std::unique_ptr<CUevent_st, EventDeleter>{event};
}

const char* _plogsubdesc = "default";

static struct hw_impl_nvmedialogenv _nvmedialogenv =
{
	.binit = 0,
};

static s32 hw_nvmedia_impl_log_init()
{
	if (HW_LIKELY(_nvmedialogenv.binit == 0))
	{
		s32 ret;
		ret = hw_plat_logcontext_fill_bydefault(&_nvmedialogenv.logcontext);
		if (ret < 0) {
			return -1;
		}
		strcpy(_nvmedialogenv.logcontext.logoper.innerimpl.logdir, "./hallog/nvmedia/multiipc_");
#if (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_MULTIIPC_PRODUCER)
		_plogsubdesc = const_cast<char*>("main");
#elif (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_MULTIIPC_CONSUMER_CUDA)
		_plogsubdesc = const_cast<char*>("cuda");
#endif
		strcat(_nvmedialogenv.logcontext.logoper.innerimpl.logdir, _plogsubdesc);
		u32 initvalue = 0;
		_nvmedialogenv.logcontext.level = HW_LOG_LEVEL_ERR;
		ret = hw_plat_logcontext_fill_bufmode_logbuf(&_nvmedialogenv.logcontext,
			_nvmedialogenv.logringbuffer, HW_NVMEDIA_IMPL_LOGRINGBUFFER_BYTECOUNT, HW_PLAT_LOGCONTEXT_LOGBUFLEVEL_DEFAULT,
			&_nvmedialogenv.atomic_offset, &initvalue);
		if (ret < 0) {
			return -1;
		}
		ret = hw_plat_logcontext_init(&_nvmedialogenv.logcontext);
		if (ret < 0) {
			return -1;
		}
		_nvmedialogenv.binit = 1;
		return 0;
	}
	return -1;
}

struct hw_plat_logcontext_t* internal_get_plogcontext_nvmedia()
{
	if (HW_UNLIKELY(_nvmedialogenv.binit == 0))
	{
		hw_nvmedia_impl_log_init();
	}
	return &_nvmedialogenv.logcontext;
}