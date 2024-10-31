/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: V2X PKI comm regist api
 * Create: 2020/09/08
 */
#ifndef V2X_COMM_REG_API_H
#define V2X_COMM_REG_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get方法提供URL 和 payload数据，返回sendId 用于接收响应数据
typedef int32_t (*CommDoGetFunc)(const char *url, uint8_t *payloadData, size_t payloadDataLen, uint32_t *sendId);
// Post方法提供URL 和 payload数据，返回sendId 用于接收响应数据
typedef int32_t (*CommDoPostFunc)(const char *url, uint8_t *payloadData, size_t payloadDataLen, uint32_t *sendId);
// 接收Get/Post的响应数据（payload），isContinue代表数据是否取完, resvDataLen为写入buffer长度
typedef int32_t (*CommRecvRespFunc)(uint32_t sendId, uint8_t *buffer, size_t bufferLen, size_t *writeBufferLen,
                                    bool *isContinue);

typedef struct {
    CommDoGetFunc doGetFunc;
    CommDoPostFunc doPostFunc;
    CommRecvRespFunc recvRespFunc;
} CommAdaptorHandleFunc;

#ifdef __cplusplus
}
#endif

#endif // V2X_COMM_REG_API_H