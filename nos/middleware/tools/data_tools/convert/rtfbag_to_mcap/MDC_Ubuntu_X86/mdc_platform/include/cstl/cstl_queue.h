/**
 * @file cstl_queue.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_queue 队列对外头文件
 * @details cstl 队列定义
 * uintptr_t
 *                          循环队列
 *       -------------<--------<-------<-------------|
 *       |    _____________________________________  |
 *       |   |   |   |   |   |   |   |   |   |   |   |
 *       |-->|   |   |   |   |   |   |   |   |   |-->|
 *           |___|___|___|___|___|___|___|___|___|
 *          head |                    |          tail
 *               |                    |
 *               | uintptr_t          |uintptr_t
 *               |--------->_______   |---------->_______
 *                           |     |               |     |
 *                           |data1|               |data2|
 *                           |_____|               |_____|
 * @date 2021-05-14
 * @version v0.1.0
 * *******************************************************************************************
 * @par 修改日志：
 * <table>
 * <tr><th>Date        <th>Version  <th>Description
 * <tr><td>2021-05-14  <td>0.1.0    <td>创建初始版本
 * </table>
 * *******************************************************************************************
 */

/**
 * @defgroup cstl_queue
 * @ingroup cstl
 */
#ifndef CSTL_QUEUE_H
#define CSTL_QUEUE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "cstl_public.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @ingroup cstl_queue
 * cstl_queue控制块
 */
typedef struct TagCstlQueue CstlQueue;

/**
 * @ingroup cstl_queue
 * @brief 创建queue。
 * @par 描述：创建一个queue并返回其控制块。
 * @attention 注意：queue不支持动态伸缩
 * @param  itemCap    [IN]  queue成员容量
 * @param  dataFunc   [IN]  data拷贝及释放函数对。
 * @retval 指向queue控制块的指针，NULL表示创建失败。
 * @par 依赖：无
 * @li cstl_queue.h：该接口声明所在的文件。
*/
CstlQueue *CstlQueueCreate(size_t itemCap, CstlDupFreeFuncPair *dataFunc);

/**
 * @ingroup cstl_queue
 * @brief 检查queue是否为空
 * @param que   [IN] queue控制块
 * @retval #true  1，queue为空。
 * @retval #false 0，queue不为空。
 * @li cstl_queue.h：该接口声明所在的文件。
 */
bool CstlQueueEmpty(const CstlQueue *que);

/**
 * @ingroup cstl_queue
 * @brief 获取queue中成员个数
 * @param que  [IN] queue控制块
 * @retval queue成员个数
 * @li cstl_queue.h：该接口声明所在的文件。
 */
size_t CstlQueueSize(const CstlQueue *que);

/**
 * @ingroup cstl_queue
 * @brief 把数据插入到queue尾。
 * @par 描述：把数据插入到queue尾。如果que满，则插入失败。
 * @param que      [IN] queue控制块。
 * @param data     [IN] 用户数据。
 * @param dataSize [IN] 用户数据拷贝长度，如果用户没有注册dupFunc，此参数将不被使用
 * @retval #CSTL_OK     插入数据成功。
 * @retval #CSTL_ERROR   插入数据失败。
 * @par 依赖：无
 * @li cstl_queue.h：该接口声明所在的文件。
*/
int32_t CstlQueuePush(CstlQueue *que, uintptr_t data, size_t dataSize);

/**
 * @ingroup cstl_queue
 * @brief 从queue头移除数据。
 * @par 描述：删除queue头数据，并移动que头到下一个用户数据位置。如果在初始化时注册了free函数，还会调该钩子函数释放用户资源。
 * @param que      [IN]    queue控制块。
 * @retval #CSTL_OK      弹出数据成功。
 * @retval #CSTL_ERROR   弹出数据失败。
 * @par 依赖：无
 * @li cstl_queue.h：该接口声明所在的文件。
 */
int32_t CstlQueuePop(CstlQueue *que);

/**
 * @ingroup cstl_queue
 * @brief 访问queue头节点，返回头节点的用户数据。用户使用前需要对队列判空，无法识别0是用户数据还是queue为空。
 * @param que [IN] queue控制块
 * @retval 头节点的用户数据。如果queue为空，则返回0。
 * @li cstl_queue.h：该接口声明所在的文件。
 */
uintptr_t CstlQueueFront(CstlQueue *que);

/**
 * @ingroup cstl_queue
 * @brief 访问queue尾节点，返回尾节点的用户数据。用户使用前需要对队列判空，无法识别0是用户数据还是queue为空。
 * @param que  [IN] queue控制块
 * @retval 尾节点的用户数据。如果queue为空，则返回0。
 * @li cstl_queue.h：该接口声明所在的文件。
 */
uintptr_t CstlQueueBack(CstlQueue *que);

/**
 * @ingroup cstl_queue
 * @brief 销毁queue
 * @attention \n
 *  1.该接口会销毁queue资源控制块，释放所有资源 \n
 *  2.如果在初始化时注册了free函数，还会调该钩子函数释放用户资源 \n
 * @par 描述：完全销毁queue
 * @param  que [IN] queue句柄。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_queue.h：该接口声明所在的文件。
 */
void CstlQueueDestroy(CstlQueue *que);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CSTL_QUEUE_H */