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

#include <iostream>
#include <sharedCtx.h>
#include <sys/types.h>
#include <tools.h>
#include <unistd.h>
#include <zmq_u.hpp>

 #ifndef checkCudaErrorsDRV
 #define checkCudaErrorsDRV(err) __checkCudaErrorsDRV(err, __FILE__, __LINE__)

 // These are the inline versions for all of the SDK helper functions
 inline void __checkCudaErrorsDRV(CUresult err, const char *file, const int line)
 {
     if (CUDA_SUCCESS != err)
     {
         const char *errorStr = NULL;
         cuGetErrorString(err, &errorStr);
         fprintf(stderr,
                 "checkCudaErrorsDRV() Driver API error = %04d \"%s\" from file <%s>, "
                 "line %i.\n",
                 err, errorStr, file, line);
         exit(EXIT_FAILURE);
     }
 }
 #endif

#define ACTIVETHREADPERCENTAGE 100.0f
struct MPS_resource
{
    CUetblSharedCtx *                    etblSharedCtxFunc = NULL;
    CU_ETBL_SHARED_CONTEXT_CREATE_PARAMS createParams      = {0};
    CUetblSharedCtx_deviceKey            deviceKey;
    CUetblSharedCtx_shareKey             shareKey;
};

bool init_mps_server(CUcontext &cuda_context, MPS_resource &mps_resource)
{

    if (CUDA_SUCCESS != cuGetExportTable((const void **)&(mps_resource.etblSharedCtxFunc), &CU_ETID_SHARED_CONTEXT))
    {
        HW_NETA_LOG_UNMASK("Shared context ETBL not found\n");
        return false;
    }
    if (CUDA_ERROR_SYSTEM_DRIVER_MISMATCH ==
        mps_resource.etblSharedCtxFunc->etiDeviceGetKey(&(mps_resource.deviceKey), 0))
    {
        HW_NETA_LOG_UNMASK("Incompatible cuCompat layer\n");
        return false;
    }
    mps_resource.createParams.activeThreadsPercentage = ACTIVETHREADPERCENTAGE;
    checkCudaErrorsDRV(
        mps_resource.etblSharedCtxFunc->etiSharedCtxCreate(&cuda_context, 0, &(mps_resource.createParams)));
    return true;
}

bool init_mps_client(zmq::socket_t &socket, CUcontext &cuda_context)
{
    // mps variables
    CUetblSharedCtx *                    etblSharedCtxFunc = NULL;
    CU_ETBL_SHARED_CONTEXT_CREATE_PARAMS createParams      = {0};
    CUetblSharedCtx_deviceKey            deviceKey;
    CUetblSharedCtx_shareKey             shareKey;

    checkCudaErrorsDRV(cuGetExportTable((const void **)&etblSharedCtxFunc, &CU_ETID_SHARED_CONTEXT));
    checkCudaErrorsDRV(etblSharedCtxFunc->etiDeviceGetKey(&deviceKey, 0));

    zmq::message_t client_deviceKey_message(&deviceKey, sizeof(CUetblSharedCtx_deviceKey));
    socket.send(client_deviceKey_message);
    zmq::message_t reply_sharedKey_message_from_server;
    socket.recv(&reply_sharedKey_message_from_server);
    if (reply_sharedKey_message_from_server.size() == 20)
    { // that means server reply error str to client
        HW_NETA_LOG_UNMASK("client recieve error message from server: %s\n", reply_sharedKey_message_from_server.data());
        return false;
    }
    memcpy(&shareKey, (void *)reply_sharedKey_message_from_server.data(), sizeof(CUetblSharedCtx_shareKey));

    createParams.activeThreadsPercentage = ACTIVETHREADPERCENTAGE;
    createParams.shareKey                = &shareKey;
    createParams.flags                   = CU_CTX_SCHED_BLOCKING_SYNC;
    HW_NETA_LOG_DEBUG(" creating a mps sub-context \n");
    checkCudaErrorsDRV(etblSharedCtxFunc->etiSharedCtxCreate(&cuda_context, 0, &createParams));

    return true;
}

bool send_shareKey(zmq::socket_t &socket, zmq::message_t &request_message, MPS_resource &mps_resource,
                   CUcontext &cuda_context)
{
    // create a shared key
    memcpy(&(mps_resource.deviceKey), (void *)request_message.data(), sizeof(CUetblSharedCtx_deviceKey));
    checkCudaErrorsDRV(mps_resource.etblSharedCtxFunc->etiSharedCtxKeyCreate(&(mps_resource.shareKey),
                                                                             mps_resource.deviceKey, cuda_context));
    zmq::message_t reply_message(&(mps_resource.shareKey), sizeof(CUetblSharedCtx_shareKey));
    socket.send(reply_message);
    return true;
}
