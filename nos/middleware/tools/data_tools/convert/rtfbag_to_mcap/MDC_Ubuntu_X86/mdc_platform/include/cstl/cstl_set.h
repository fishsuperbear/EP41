/**
 * @file cstl_set.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_set 对外头文件
 * @details cstl_set 集合对外头文件
 * @date 2021-04-15
 * @version v0.1.0
 * *******************************************************************************************
 * @par 修改日志：
 * <table>
 * <tr><th>Date        <th>Version  <th>Description
 * <tr><td>2021-04-15  <td>0.1.0    <td>创建初始版本
 * </table>
 * *******************************************************************************************
 */

/**
 * @defgroup cstl_set set表
 * @ingroup cstl
 */
#ifndef CSTL_SET_H
#define CSTL_SET_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "cstl_map.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @ingroup cstl_set
 * cstl_set控制块
 */
typedef CstlMap CstlSet;

/**
 * @ingroup cstl_set
 * set迭代器定义，指向用户数据（key）的起始地址
 */
typedef CstlMapIterator CstlSetIterator;

/**
 * @ingroup cstl_set
 * @brief 创建set。
 * @par 描述：创建一个set，并返回其控制块指针。
 * @attention
 * 1.set存储的key为uintptr_t类型，该值可以为用户key，也可是指向用户key的地址。
 * 2.用户未注册keyCmpFunc函数时，将使用默认比较函数，key会被默认为uintptr_t型。
 * 3.当用户key为字符串或其他复杂类型时，应将其地址作为参数（需强转为uintptr_t类型），\n
    若该地址指向栈内存，建议注册keyFunc->dupFunc进行key的拷贝，返回对应的堆内存地址。
 * 4.若用户key为字符串或其他复杂类型，释放set表或擦除节点时，用户需自行释放资源或注册keyFunc->freeFunc。
 * @param  keyCmpFunc   [IN]  key的比较函数。
 * @param  keyFunc      [IN]  key拷贝函数及释放函数对。
 * @retval 指向set控制块的指针，NULL表示创建失败。
 * @par 依赖：无。
 * @li cstl_set.h：该接口声明所在的文件。
 */
CstlSet *CstlSetCreate(CstlKeyCmpFunc keyCmpFunc, CstlDupFreeFuncPair *keyFunc);

/**
 * @ingroup cstl_set
 * @brief 插入set数据
 * @par 描述：插入用户数据
 * @attention
 * 1.不支持重复的key。
 * 2.key为int类型时，直接将值作为参数即可。
 * 3.当用户key为字符串或其他复杂类型时，应将其地址作为参数（需强转为uintptr_t类型），
     如char *key, 则入参应为(uintptr_t)key。
 * @param set       [IN] set句柄
 * @param key       [IN] key或指向key的地址
 * @param keySize   [IN] key拷贝长度，若用户未注册dupFunc，此参数将不被使用
 * @retval CSTL_OK 插入成员成功
 * @retval CSTL_ERROR 插入成员失败
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
int32_t CstlSetInsert(CstlSet *set, uintptr_t key, size_t keySize);

/**
 * @ingroup cstl_set
 * @brief 检查Set是否为空
 * @param set    [IN] set控制块
 * @retval #true  1，Set为空。
 * @retval #false 0，Set不为空。
 * @li cstl_set.h：该接口声明所在的文件。
 */
bool CstlSetEmpty(const CstlSet *set);

/**
 * @ingroup cstl_set
 * @brief 获取set中成员个数
 * @param set  [IN] set控制块
 * @retval set成员个数
 * @li cstl_set.h：该接口声明所在的文件。
 */
size_t CstlSetSize(const CstlSet *set);

/**
 * @ingroup cstl_set
 * @brief 查找成员
 * @par 描述: 根据key查找并返回迭代器
 * @param set       [IN] set句柄
 * @param key       [IN] key或指向key的地址
 * @retval #key对应的迭代器，失败时返回CstlSetIterEnd()
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
CstlSetIterator CstlSetFind(const CstlSet *set, uintptr_t key);

/**
 * @ingroup cstl_set
 * @brief 从set中移除指定成员，并返回下一个节点的迭代器
 * @par 描述: 根据key查找到成员并删除（释放内存）
 * @param set      [IN] set句柄
 * @param key      [IN] 待移除的成员key。
 * @retval  下一个节点迭代器。如果被删除key为最后一个节点，则返回CstlSetIterEnd()
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
CstlSetIterator CstlSetErase(CstlSet *set, uintptr_t key);

/**
 * @ingroup cstl_set
 * @brief 获取set第一个成员迭代器。
 * @par 描述：获取set第一个成员迭代器。
 * @param set [IN] set句柄。
 * @retval #第一个迭代器。当set中无节点时，返回CstlSetIterEnd()。
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
CstlSetIterator CstlSetIterBegin(const CstlSet *set);

/**
 * @ingroup cstl_set
 * @brief 获取set下一个成员迭代器。
 * @par 描述：获取set下一个成员迭代器。
 * @attention
 * @param set  [IN]  set句柄。
 * @param it   [IN]  当前迭代器。
 * @retval #下一个成员迭代器。
 * @retval #CstlSetIterEnd()，表示当前已是最后一个节点。
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
CstlSetIterator CstlSetIterNext(const CstlSet *set, CstlSetIterator it);

/**
 * @ingroup cstl_set
 * @brief 获取set最后一个节点之后预留的迭代器。
 * @par 描述：获取set最后一个节点之后预留的迭代器。
 * @param set  [IN]  set句柄。
 * @retval #最后一个节点之后预留的迭代器。
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
CstlSetIterator CstlSetIterEnd(const CstlSet *set);

/**
 * @ingroup cstl_set
 * @brief 获取当前迭代器的key。
 * @par 描述：获取当前迭代器的key。
 * @param it  [IN]  当前迭代器。
 * @retval 迭代器对应的key。
 * @retval #0 it等于CstlSetIterEnd()
 * @par 依赖：无
 * @attention 当set为空指针或it等于#CstlSetIterEnd()时，接口返回0，该接口无法区分是错误码还是用户数据，
 * 用户调用该接口时必须保证set为合法指针，并且it不等于#CstlSetIterEnd()
 * @li cstl_set.h：该接口声明所在的文件。
 */
uintptr_t CstlSetIterKey(const CstlSet *set, CstlSetIterator it);

/**
 * @ingroup cstl_set
 * @brief 删除set所有成员
 * @attention
 *  1.该接口会删除set表中所有节点。
 *  2.若用户注册了key的资源释放函数，则会调用钩子进行用户资源释放。
 * @par 描述：删除所有成员（set还在，只是没有成员）
 * @param  set [IN] set句柄。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
void CstlSetClear(CstlSet *set);

/**
 * @ingroup cstl_set
 * @brief 删除set
 * @attention
 *  1.该接口会删除set表中所有节点。
 *  2.若用户注册了key的资源释放函数，则会调用钩子进行用户资源释放。
 *  3.调用本接口后，set指针指向的内存被free，set表不可以再次访问。
 * @par 描述：删除set，如里面有成员先删除成员，再回收控制块内存。
 * @param  set [IN] set句柄。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_set.h：该接口声明所在的文件。
 */
void CstlSetDestory(CstlSet *set);


#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CSTL_SET_H */

