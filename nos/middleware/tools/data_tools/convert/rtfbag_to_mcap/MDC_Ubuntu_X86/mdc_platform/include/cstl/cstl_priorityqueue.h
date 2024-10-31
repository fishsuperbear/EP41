/**
 * @file cstl_priorityqueue.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_priorityqueue 对外头文件
 * @details 优先队列定义
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
 * @defgroup cstl_priorityqueue 优先队列
 * @ingroup cstl
 */

#ifndef CSTL_PRIORITYQUEUE_H
#define CSTL_PRIORITYQUEUE_H

#include <stdbool.h>
#include "cstl_public.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @ingroup cstl_priorityqueue
 * @brief 比较函数原型
 * @par 描述：比较函数原型，用于排序场景。
 * @attention \n
 *  1.这里只定义了比较函数原型，由于不知道数据类型和长度，因此钩子函数需要业务自己实现。\n
 *  2.对于data1和data2，比较函数不区分是整型数还是指针，用户应自己区分。\n
 *  3.用户返回(data1 - data2)的时候为大顶堆，反之为小顶堆。
 * @param data1    [IN] 用户需要插入的数据
 * @param data2    [IN] 队列中待比较的数据
 * @retval 大于0, 等于0, 小于0
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
typedef int32_t (*CstlPriQueCmpFunc)(uintptr_t data1, uintptr_t data2);

/**
 * @ingroup cstl_priorityqueue
 * cstl_priorityqueue 控制块
 */
typedef struct tagCstlPriQue CstlPriQue;

/**
 * @ingroup cstl_priorityqueue
 * @brief 创建priority queue。
 * @par 描述：创建一个priority queue，并返回其控制块指针。
 * @attention \n
 *  1.提供默认的整型及字符串型比较函数#CstlIntCmpFunc、#CstlStrCmpFunc，用户需要在创建队列时作为参数传入。\n
 *  2.默认比较函数均为大顶堆，用户需要小顶堆需要自定义比较函数。\n
 *  3.如果扩展数据的生命周期小于节点的生命周期，则需要在创建队列时注册dataFunc->dupFunc。\n
 *  4.用户使用# CstlPriQuePushBatch添加数据时，优先队列无法感知用户数据类型，必须注册注册dataFunc->dupFunc。
 * @param  cmpFunc      [IN] 比较函数。
 * @param  dataFun      [IN] 用户数据拷贝及释放函数对。
 * @retval #指向priority queue控制块的指针 创建成功。
 * @retval #NULL 创建失败。
 * @par 依赖：无。
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
CstlPriQue *CstlPriQueCreate(CstlPriQueCmpFunc cmpFunc, CstlDupFreeFuncPair *dataFunc);

/**
 * @ingroup cstl_priorityqueue
 * @brief 插入一个数据到priority queue。
 * @par 描述：用户数据插入后，调用钩子cmpFunc完成优先队列转换。
 * @attention \n
 *  1.用户数据。用户应清晰的知道value是整型数还是地址。\n
 *  2.当用户存储的数据长度不大于sizeof(uintptr_t)时，直接存储数据即可，无需申请内存。\n
 *  3.当用户存储的数据长度大于sizeof(uintptr_t)时，用户需要传入数据的地址。\n
 *  4.如果扩展数据的生命周期小于节点的生命周期，则需要在创建队列时注册dataFunc->dupFunc。\n
 * @param priQueue       [IN] priority queue控制块。
 * @param value          [IN] 用户保存的数据
 * @param valueSize      [IN] 用户保存的数据的拷贝长度，如果用户没有注册dupFunc，该参数将不被使用
 * @retval #CSTL_OK      成功。
 * @retval #CSTL_ERROR   失败。
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
int32_t CstlPriQuePush(CstlPriQue *priQueue, uintptr_t value, size_t valueSize);

/**
 * @ingroup cstl_priorityqueue
 * @brief 插入一组数据到priority queue。
 * @par 描述：插入一组数据到priority queue，并转换成优先队列。
 * @attention \n
 *  1.从起始地址startAddr，按照单个数据长度为itemSize进行读取，共读取itemNum个。\n
 *  2.用户必须在创建队列时注册拷贝函数(dataFunc.dupFunc)进行内容拷贝，并返回指向保存用户数据的地址。\n
 *  3.若用户保存的数据类型为整型，可以在拷贝函数(dataFunc.dupFunc)中返回该整型值。\n
 *  4.用户数据插入后，会调用钩子cmpFunc完成排序。\n
 * @param priQueue       [IN] priority queue控制块。
 * @param beginItemAddr  [IN] 指向用户数据的起始地址。
 * @param itemNum        [IN] 待插入的数据个数。
 * @param itemSize       [IN] 单个数据的大小。
 * @param dupSize        [IN] 每个数据的拷贝长度，用做dupFunc函数参数，当用户没注册dupFunc时，该参数将不被使用
 * @retval #CSTL_OK       成功。
 * @retval #CSTL_ERROR    失败。
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
int32_t CstlPriQuePushBatch(CstlPriQue *priQueue, const void *beginItemAddr,
                            size_t itemNum, size_t itemSize, size_t dupSize);

/**
 * @ingroup cstl_priorityqueue
 * @brief 弹出头部数据。
 * @par 描述：弹出头部数据，并释放资源。
 * @attention \n
 *  1.如果用户在创建队列时注册了dataFunc->freeFunc，则会调用该钩子释放用户资源。\n
 *  2.建议用户dataFunc->dupFunc与dataFunc->freeFunc函数成对注册。\n
 * @param priQueue    [IN]    priority queue控制块。
 * @retval 无。
 * @par 依赖：无。
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
void CstlPriQuePop(CstlPriQue *priQueue);

/**
 * @ingroup cstl_priorityqueue
 * @brief 读取头数据。
 * @par 返回堆顶的数据。
 * @attention \n
 *  1.返回的数据是最大值还是最小值，由用户创建优先队列时注册的比较函数决定。\n
 *  2.当队列为空时，接口返回0，该接口无法识别返回的值是错误码还是用户数据，
      用户使用前必须进行#CstlPriQueEmpty判断是否为空。\n
 * @param priQueue    [IN]    priority queue控制块。
 * @retval 用户数据。
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
uintptr_t CstlPriQueTop(const CstlPriQue *priQueue);

/**
 * @ingroup cstl_priorityqueue
 * @brief 判断priority queue是否为空
 * @param priQueue  [IN] priority queue控制块。
 * @retval #true，  表示priority queue为空
 * @retval #false， 表示priority queue为非空
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
bool CstlPriQueEmpty(const CstlPriQue *priQueue);

/**
 * @ingroup cstl_priorityqueue
 * @brief 获取priority queue中成员个数
 * @param priQueue  [IN] priority queue控制块。
 * @retval priority queue成员个数
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
size_t CstlPriQueSize(const CstlPriQue *priQueue);

/**
 * @ingroup cstl_priorityqueue
 * @brief 删除priority queue所有成员，保留控制块
 * @par 描述：该接口调用后，控制块可以继续使用。
 * @attention \n
 *  1.调用该接口后priQueue指向的内存被释放。\n
 *  2.如果用户注册了释放函数，会调用该函数进行用户资源释放。\n
 * @param  priQueue [IN] priority queue控制块。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
void CstlPriQueClear(CstlPriQue *priQueue);

/**
 * @ingroup cstl_priorityqueue
 * @brief 删除priority queue所有成员及控制块
 * @par 描述：销毁priority queue。
 * @attention \n
 *  1.调用该接口后priQueue指向的内存被释放。\n
 *  2.如果用户注册了释放函数，会调用该函数进行用户资源释放。\n
 *  3.该接口调用后句柄被释放，不允许再次使用。
 * @param  priQueue [IN] priority queue控制块。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_priorityqueue.h：该接口声明所在的文件。
 */
void CstlPriQueDestory(CstlPriQue *priQueue);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CSTL_PRIORITY_QUEUE_H */

