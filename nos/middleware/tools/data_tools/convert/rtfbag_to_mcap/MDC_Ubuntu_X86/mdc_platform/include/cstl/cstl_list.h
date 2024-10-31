/**
 * @file cstl_list.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_list 对外头文件
 * @details  1、本头文件是在 cstl_rawlist.h 基础上，借鉴C++面向对象抽象方法，从方便用户使用的角度出发，抽象提炼的接口。
 *             用户不需要看到链表节点，只关注自己的数据。
 *           2、关于 duplicate 函数
 *             a）duplicate 函数在初始化时注册；
 *             b）如果用户不注册 duplicate 函数，则表示用户的数据类型是 int 型的，且长度 <= sizeof(uintptr_t)，
 *             如32位环境下uint32_t、64位环境下uint32_t、uint64_t等；
 *             c）对于其它情况，用户都必须注册 duplicate 函数。
 *           3、关于free函数
 *             a）如果用户数据是int型、且长度<=sizeof(uintptr_t)，则用户不需要注册资源释放函数；
 *             b）其它场景，用户都必须在初始化时注册free函数。
 *           4、链表处理内部无锁，如果用户需要支持多线程并发，则需要在外层加锁。
 *   +------------------------------------------------------------------------+
 *   | +-------------------------------------------------------------------+  |
 *   | |       head                                                        |  |
 *   | |   +----------+    +-------------------------+     +-----------+   |  |
 *   | +---|   prev   |<---| prev                    |<----| prev      |<--+  |
 *   +---->|   next   |--->| next                    |---->| next      |------+
 *         +----------+    +-------------------------+     +-----------+
 *         | count    |    | userdata    int型且     |     | userdata  |
 *         | freefunc |    | 长度<=sizeof(uintptr_t) |     | (pointer) |--------+  用户私有资源
 *         | dupfunc  |    +-------------------------+     +-----------+       \|/
 *         +----------+                                                  +-------------+
 *                                                                       | privatedata |
 *                                                                       +-------------+
 * @date 2021-05-14
 * @version v0.1.0
 * *******************************************************************************************
 * @par 修改日志：
 * <table>
 * <tr><th>Date        <th>Version  <th>Description
 * <tr><td>2021-05-14  <td>0.1.0    <td>初始化版本
 * </table>
 * *******************************************************************************************
 * @par 修改日志：
 * <table>
 * <tr><th>Date        <th>Version  <th>Description
 * <tr><td>2021-07-20  <td>1.0.0    <td>车规规范整改
 * </table>
 * *******************************************************************************************
 */

/**
 * @defgroup cstl_list 双向链表
 * @ingroup cstl
 */
#ifndef CSTL_LIST_H
#define CSTL_LIST_H

#include <stdint.h>

#include "cstl_rawlist.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cstl_list
 * 链表头
 */
typedef struct {
    CstlRawList rawList;
    CstlDupFreeFuncPair dataFunc;
} CstlList;

/**
 * @ingroup cstl_list
 * 链表迭代器定义
 */
typedef struct TagCstlListNode *CstlListIterator;

/**
 * @ingroup cstl_list
 * @brief 初始化链表
 * @par 描述：初始化链表，按需注册用户数据dup函数和用户数据资源free函数。\n
 * @attention \n
 *  1.如果用户想要存储的数据为整型，且长度<=sizeof(uintptr_t)，则无需注册dataFunc.dupFunc，赋空即可。\n
 *  2.如果用户数据为字符串或其他自定义复杂数据类型，且数据生命周期小于节点生命周期，则必须注册dataFunc->dupFunc进行数据拷贝。
 * @param list       [IN] 链表
 * @param dataFunc   [IN] 用户数据拷贝及释放函数对，如果用户未注册dataFunc.dupFunc，则默认data为整型。
 * @retval #CSTL_OK  0，链表初始化成功。
 * @retval #ERRNO_CSTL_INPUT_INVALID，即0xa030002，表明链表为NULL，初始化失败。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListInit(CstlList *list, const CstlDupFreeFuncPair *dataFunc);

/**
 * @ingroup cstl_list
 * @brief 链表节点清空，删除所有节点
 * @par 描述：链表节点清空，删除所有节点，调用户注册的free函数释放用户资源，回归链表初始化后的状态。
 * @param list    [IN]  链表
 * @retval #CSTL_OK  0，链表清空成功。
 * @retval #ERRNO_CSTL_INPUT_INVALID，即0xa030002，表明链表为NULL，清空失败。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListClear(CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 链表去初始化
 * @par 描述：链表去初始化：删除所有节点，调用户注册的free函数释放用户资源，去注册钩子函数。
 * @param list        [IN]  链表
 * @retval #CSTL_OK  0，链表去初始化成功。
 * @retval #ERRNO_CSTL_INPUT_INVALID，即0xa030002，表明链表为NULL，去初始化失败。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListDeinit(CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 检查链表是否为空
 * @param list    [IN] 待检查的链表
 * @retval #true  1，链表为NULL或者无数据。
 * @retval #false 0，链表不为空。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
bool CstlListEmpty(const CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 获取链表中节点个数
 * @param list  [IN] 链表
 * @retval 链表节点个数
 * @li cstl_list.h：该接口声明所在的头文件。
 */
size_t CstlListSize(const CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 在链表头插入用户数据。
 * @param list         [IN] 链表
 * @param userData     [IN] 待插入的数据或指向用户私有数据的指针
 * @param userDataSize [IN] 数据拷贝长度，若用户未注册dupFunc，该参数将不被使用
 * @retval #CSTL_OK     插入数据成功
 * @retval #CSTL_ERROR  函数执行中内存拷贝存在错误
 * @retval #ERRNO_CSTL_INPUT_INVALID，即0xa030002，表明入参list是NULL
 * @retval #ERRNO_CSTL_NODE_CREATE_FAIL，即0xa030003，创建新的节点失败，插入失败
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListPushFront(CstlList *list, uintptr_t userData, size_t userDataSize);

/**
 * @ingroup cstl_list
 * @brief 在链表尾插入用户数据
 * @param list         [IN] 链表
 * @param userData     [IN] 待插入的数据或指向用户私有数据的指针
 * @param userDataSize [IN] 数据拷贝长度，若用户未注册dupFunc，该参数将不被使用
 * @retval #CSTL_OK     插入数据成功
 * @retval #CSTL_ERROR  函数执行中内存拷贝存在错误
 * @retval #ERRNO_CSTL_INPUT_INVALID，即0xa030002，表明入参list是NULL
 * @retval #ERRNO_CSTL_NODE_CREATE_FAIL，即0xa030003，创建新的节点失败，插入失败
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListPushBack(CstlList *list, uintptr_t userData, size_t userDataSize);

/**
 * @ingroup cstl_list
 * @brief 从链表头部POP一个节点
 * @par 描述：从链表中移除头节点，同时释放节点内存。如果在初始化时注册了free函数，还会调该钩子函数释放用户私有资源。\n
 * 如果链表为空，则不做任何事情。
 * @param list [IN]  链表
 * @retval #CSTL_OK  0，弹出头部成功。
 * @retval #ERRNO_CSTL_ELEMENT_EMPTY，即0xa030001。表明链表为NULL，或者链表无数据
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListPopFront(CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 从链表尾部POP一个节点。
 * @par 描述：从链表中移除尾节点，同时释放节点内存。如果在初始化时注册了free函数，还会调该钩子函数释放用户私有资源。\n
 * 如果链表为空，则不做任何事情。
 * @param list [IN]  链表
 * @retval #CSTL_OK  0，弹出尾部成功。
 * @retval #ERRNO_CSTL_ELEMENT_EMPTY，即0xa030001,表明链表为NULL，或者链表无数据
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListPopBack(CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 访问链表头节点，返回头节点的用户数据。
 * @par 描述：访问链表头节点，返回头节点的用户数据
 * @attention 注意：如果链表为空，则不能区分是因为链表为空返回0，还是返回的真实数据就是0。\n
 * 因此用户在调用本函数前必须先判链表是否为空。
 * @param list [IN] 链表
 * @retval 头节点的用户数据/指针。如果链表为空，则返回0。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
uintptr_t CstlListFront(const CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 访问链表尾节点，返回尾节点的用户数据。
 * @attention 注意：如果链表为空，则不能区分是因为链表为空返回0，还是返回的真实数据就是0。\n
 * 因此用户在调用本函数前必须先判链表是否为空。
 * @param list  [IN] 链表
 * @retval 尾节点的用户数据/指针。如果链表为空，则返回0。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
uintptr_t CstlListBack(const CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 获取链表头节点迭代器。
 * @param list [IN] 链表
 * @retval 链表头节点迭代器。如果链表为空，则指向链表头。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
CstlListIterator CstlListIterBegin(const CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 获取链表尾节点的下一个节点迭代器
 * @param list  [IN] 链表
 * @attention 如果用户传入的list为NULL，则一定返回NULL，所以用户需要使用正确的参数。
 * @retval 链表尾节点的下一个节点迭代器（指向链表头）。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
CstlListIterator CstlListIterEnd(CstlList *list);

/**
 * @ingroup cstl_list
 * @brief 获取上一个节点迭代器
 * @param list [IN] 链表
 * @param it   [IN] 迭代器
 * @attention 如果用户传入的list为NULL或者it不是list的合法部分，则一定返回NULL，所以用户需要使用正确的参数。
 * @retval list非空，上一个节点迭代器。
 * @retval list是NULL，返回NULL。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
CstlListIterator CstlListIterPrev(const CstlList *list, const CstlListIterator it);

/**
 * @ingroup cstl_list
 * @brief 获取下一个节点迭代器
 * @param list [IN] 链表
 * @param it   [IN] 迭代器
 * @attention 如果用户传入的list为NULL或者it不是list的合法部分，则一定返回NULL，所以用户需要使用正确的参数。
 * @retval 非空时返回下一个节点迭代器。
 * @retval list是NULL，返回NULL。
 * @li cstl_list.h：该接口声明所在的头文件。
 */
CstlListIterator CstlListIterNext(const CstlList *list, const CstlListIterator it);

/**
 * @ingroup cstl_list
 * @brief 在指定迭代器指向的节点前插入数据。
 * @param list         [IN] 链表
 * @param it           [IN] 当前迭代器位置
 * @param userData     [IN] 待插入的数据或指向用户私有数据的指针
 * @param userDataSize [IN] 数据拷贝长度，若用户未注册dupFunc，该参数将不被使用
 * @retval #CSTL_OK     插入数据成功
 * @retval #CSTL_ERROR  函数执行中内存拷贝存在错误
 * @retval #ERRNO_CSTL_INPUT_INVALID，即0xa030002表明入参不合理
 * @retval #ERRNO_CSTL_NODE_CREATE_FAIL，即0xa030003 创建新的节点失败，插入失败
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListInsert(CstlList *list, const CstlListIterator it, uintptr_t userData, size_t userDataSize);

/**
 * @ingroup cstl_list
 * @brief 删除链表中指定节点，同时释放节点内存。
 * @par 描述：从链表中删除指定节点，同时释放节点内存。\n
 * 如果在初始化时注册了free函数，还会调该钩子函数释放用户数据中的句柄、指针等私有资源。
 * @attention 如果用户传入的list为NULL或者it不是list的合法部分，则一定返回NULL，所以用户需要使用正确的参数。
 * @param list [IN] 链表
 * @param it   [IN] 待删除的节点迭代器
 * @retval 被删除节点的下一个节点迭代器，如果被删除节点是尾节点，则返回的迭代器指向链表头
 * @li cstl_list.h：该接口声明所在的头文件。
 */
CstlListIterator CstlListIterErase(CstlList *list, CstlListIterator it);

/**
 * @ingroup cstl_list
 * @brief 获取用户数据
 * @attention 调用者必须保证参数的合法性。如果入参非法，则返回0，调用者不能区分是正常数据，还有由于参数非法返回的0。
 * @param it [IN] 链表迭代器
 * @retval 用户数据
 * @li cstl_list.h：该接口声明所在的头文件。
 */
uintptr_t CstlListIterData(const CstlListIterator it);

/**
 * @ingroup cstl_list
 * @brief 根据用户定义的排序函数，对链表节点进行排序。
 * @par 描述：根据用户定义的排序函数，对链表节点进行排序，排序顺序按排序函数定义实行。
 * @attention \n
 * 1、此处用户输入的排序函数钩子，其两个入参为两个待比较的节点数据值userdata，入参类型是（uintptr_t）。
 * @param list      [IN] 链表
 * @param cmpFunc   [IN] 排序函数钩子
 * @retval CSTL_OK      排序成功
 * @retval #ERRNO_CSTL_INPUT_INVALID，即0xa030002表明入参不合理
 * @li cstl_list.h：该接口声明所在的头文件。
 */
int32_t CstlListSort(const CstlList *list, CstlKeyCmpFunc cmpFunc);

/**
 * @ingroup cstl_list
 * @brief 根据用户定义的迭代器匹配函数，搜索用户想要的迭代器，即节点指针。
 * @par 描述：根据用户定义的迭代器匹配函数，搜索用户想要的迭代器，即节点指针。
 * @attention \n
 * 1、将从链表头往后遍历，对每个节点依次调用匹配函数，直到找到第一个匹配的节点或者遍历到链表尾部结束。\n
 * 2、此处用户输入的匹配函数钩子，其第一个入参地址是每个待搜索的节点的数据值userdata，入参类型是（uintptr_t）。
 * 3、如果用户传入的list为NULL或者是比较函数是NULL则一定返回NULL，所以用户需要使用正确的参数。
 * @param list           [IN] 链表
 * @param iterCmpFunc    [IN] 匹配函数钩子
 * @param data           [IN] 数据信息
 * @retval 非NULL 查询成功，返回匹配的节点迭代器
 * @retval NULL   查询失败
 * @li cstl_list.h：该接口声明所在的头文件。
 */
CstlListIterator CstlListIterFind(CstlList *list, CstlKeyCmpFunc iterCmpFunc, uintptr_t data);

#ifdef __cplusplus
}
#endif

#endif /* CSTL_LIST_H */

