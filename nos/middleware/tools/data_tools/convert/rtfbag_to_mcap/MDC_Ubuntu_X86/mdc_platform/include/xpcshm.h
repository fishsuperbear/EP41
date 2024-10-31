/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: xpcshm header
 * Author: Huawei Godel Lab
 * Create: 2020-10-10
 */

#ifndef XPCSHM_H
#define XPCSHM_H

#include <stdint.h>
#include "xpcshm_common.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define INVALID_FD (-1)

struct ChlInfo {
    char name[XPC_CHANNEL_NAME_MAX]; // channel名字
    int32_t key; // 0 为不指定key(name方式);传入指定key(非0值), 用于下一次以key的方式打开此channel, 需要跟flag参数CHL_O_BYKEY配合
    int32_t fifoSize; // 一个FIFO的大小，(备用，填0)
    int32_t virtDomain; // 虚拟域ID,(0代表非虚拟域)
};

struct ChlPollRet {
    int32_t id;
    uint32_t type;
    uint32_t len;
};

/*
 * Open one channel memory from xpc channel.
 * ch: channel name, specify one share memory object to be created or opened.
 *    key: an alternative way to find a channel except name.
 *    fifoSize: reserve.
 *    virtDomain: for virtual domain id.
 * flag: support O_CHL_CREAT / O_CHL_EXECL/ O_CHL_OPENONLY / O_CHL_BYKEY.
 * return value >= 0: opened or created share memory id, used for other operations.
 *              <0: error reason on failure.
 */
int32_t XpcOpenChannel(const struct ChlInfo *ch, uint32_t flag);

/*
 * Close one channel opened by XpcOpenChannel.
 * chId: the channel id returned by XpcOpenChannel.
 */
void XpcCloseChannel(int32_t chId);

/*
 * Write buffer data to channel, incase to send data to domain specified.
 * chId: the channel id returned by XpcOpenChannel.
 * buf: user data container.
 * len: the length of buf.
 * return value == 0: write succ.
 *              != 0: write failed;
 */
int32_t XpcWrite(int32_t chId, const char *buf, uint32_t len);

/*
 * Read a bulk of data from channel specified.
 * chId: the channel id returned by XpcOpenChannel.
 * buf: user data container, managed by user.
 * len: the length of buf, also as a passed out param means actual write len.
 * nextType: the next data type, can be PKT_TYPE_NORMAL/PKT_TYPE_PTR.
 * return value == 0: success but no next data.
 *              < 0: error reason on failure.
 *              > 0: have next package.
 */
int32_t XpcRead(int32_t chId, char *buf, uint32_t *len, uint32_t *nextType);

int32_t XpcAlloc(int32_t chlId, uint32_t size, char **outVa);

/*
 * Request tick of the id, calling func while received one tick.
 * Note: don't block signal 52.
 * id: request id of the tick.
 * func: callback function while the tick of the id received.
 *
 */

/*
 * Unmap the address of the zero copy memory object XpcAlloc to.
 * buffer: address to unmap, got by XpcAlloc, like munmap syscall.
 */
void XpcFree(const char *buffer);

/*
 * Send the alloced buffer to peer domain, the buffer is obtained by XpcAlloc.
 * chId: the channel id returned by XpcOpenChannel.
 * buf: alloced buffer is obtained by XpcAlloc.
 * len: memcpy len in actual use.
 * return value == 0: write succ.
 *              != 0: write failed;
 */
int32_t XpcSendBuffer(int32_t chId, const char *buf, uint32_t len);

/*
 * Get a bulk of data from channel specified.
 * chId: the channel id returned by XpcOpenChannel.
 * buf: user process vma, managed by user.
 * len: the length of buf, passed out param means actual write len.
 * nextType: the next data type, can be PKT_TYPE_NORMAL/PKT_TYPE_PTR.
 * return value == 0: success but no next data.
 *              < 0: error reason on failure.
 *              > 0: have next package.
 */
int32_t XpcGetBuffer(int32_t chId, char **buf, uint32_t *len, uint32_t *nextType);

/**
 * Get a channel status of specified id.
 * chId: the channel id returned by XpcOpenChannel.
 * return value == XPC_CHL_STATE_CLOSE, means close;
                   XPC_CHL_STATE_OPENING, means opened by one domain;
                   XPC_CHL_STATE_OPEN, means complete open.
*/
int32_t XpcGetChlState(int32_t chId);

/**
 * DFX print tools function, it will print accepted package data count and bulk summary.
 */
void XpcPerformance(void);

/**
 * DFX print tools function, it will print memory usage status, used/free/total proportion.
 */
void XpcMemInfo(void);

/**
 * DFX print tools function, it will print memory usage status, used/free/total proportion.
 * chId: the channel id returned by XpcOpenChannel.
 * count: user passed this to carry out this channel accepted package data count.
 * bulk: user passed this to carry out this channel accepted package bulk summary.
 * return value == 0, means success;
                < 0, means failure;
*/
int32_t XpcChlInfo(int32_t chId, uint32_t *count, uint32_t *bulk);

/*
 * Poll multiple target channels.
 * wait until at least one of target channels has one pkt to read or timeout.
 * chId: restore the channel ids to poll.
 * chlNum: the num of chId[].
 * res: the result of poll, is an array.
 * resNum: the num of res[].
 * timeout: the time to wait for pkt (millisecond).
 *          -1 will wait until has ptk to read, no timeout.
 * return value == 0: succuss.
 *              != 0: error reason on failure.
 */
int32_t XpcPoll(const int32_t *chId, uint32_t chlNum, struct ChlPollRet *res, int32_t *resNum, int32_t timeout);


#ifndef ON_CORE
/*
 * Mmap one named memblk.
 * chlId: restore the channel ids to poll.
 * flag: indicates.
 * size: size of the memblk (bytes).
 * outVa: output the addr of the named memblk.
 * return value: == 0: success
 *               != 0: error reason on failure.
 */
int32_t XpcMmap(int32_t chlId, uint32_t flag, const char *subName, uint32_t size, char **outVa);
#endif // ON_CORE

#if defined(__cplusplus)
}
#endif

#endif
