/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description:
 * Create: 2020/8/19
 * History:
 * 2020/8/19 第一次创建
 */

/** @defgroup cme cme */
/** @defgroup cme_common CME通用接口
 *  @ingroup cme
 */
/** @defgroup cme_list CME_List_API
 * @ingroup cme_common
 */
#ifndef CME_LIST_API_H
#define CME_LIST_API_H

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include "asn1_types_api.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef CstlList CmeList;

/**
 * @ingroup cme_asn1
 * @brief 链表节点句柄
 */
typedef CstlListIterator ListNodeHandle;
typedef const CstlListIterator ListNodeRoHandle;

#ifdef __cplusplus
}
#endif

#endif
