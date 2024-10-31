/**
 * @file cstl_rawhash.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_rawhash 对外头文件
 * @details
 * Notes: 1、key和value在rawHash中连续存储，用户需要注册key和value的拷贝函数。
 *           对于key和value无特殊限制，只要不超过hash表创建是指定的keySize和valueSize即可。
 *           注意：如果key是字符串，要考虑到最后‘\0’的位置。
 *        2、用户不感知冲突数组，只需要调 CstlRawHashInsert/CstlRawHashErase 即可。
 *        3、如果用户数据中有指针，则必须在创建hash表示注册用户数据free函数，即下图中场景中的扩展数据需用户自行释放。
 *   +--------+
 *   | 控制块 |
 *   +------- +  冲突数组 +-------+
 *   |   0    | <-------> | count  |
 *   +--------+           | curCap |
 *   |   1    |           +-------+
 *   |--------+           |hashCode|
 *   |        +           |  key   |
 *   |  ...   |           |  value |
 *   +--------+           +------- +
 *                        |hashCode|
 *                        |  key   |     场景：value中有扩展数据
 *                        |  value |------>+----------+
 *                        +-------+        |  others  |
 *                                         +----------+
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
 * @defgroup cstl_rawhash 哈希表
 * @ingroup cstl
 */

#ifndef CSTL_RAWHASH_H
#define CSTL_RAWHASH_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "cstl_public.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cstl_rawhash
 * hash表句柄
 */
typedef struct TagRawHashTable CstlRawHash;

/**
 * @ingroup cstl_rawhash
 * @brief 用户数据拷贝函数原型
 * @attention 注意：用户需要将src中的数据拷贝到dest中。
 * @param dest      [IN]  目的缓冲区
 * @param destSize  [IN]  目的缓冲区长度，用户创建hash表时注册的keySize。
 * @param src       [IN]  源缓冲区
 * @param srcSize   [IN]  源缓冲区长度，用户实际keySize。
 * @retval 目标缓冲区，NULL表示失败。
 */
typedef int32_t (*CstlRawDupFunc)(void *dest, size_t destSize, const void *src, size_t srcSize);

/**
 * @ingroup cstl_rawhash
 * @brief 用户内存释放函数原型
 * @par 描述：资源释放函数原型，用户只能释放ptr中的扩展数据，不可以释放ptr自身。
 * @param ptr    [IN] 指向用户数据的指针
 * @retval 无
 */
typedef void (*CstlRawFreeFunc)(void *ptr);

/**
 * @ingroup cstl_rawhash
 * @brief key和value函数原型对
 * @par 描述：key和value的拷贝及释放函数成对出现。
 */
typedef struct {
    CstlRawDupFunc dupFunc;       /**< 复制函数 */
    CstlRawFreeFunc freeFunc;     /**< 释放函数 */
} CstlRawDupFreeFuncPair;

/**
 * @ingroup cstl_rawhash
 * @brief 该函数根据输入的key生成hash值。
 * @param key      [IN] hash key
 * @param keySize  [IN] hash key的长度
 */
typedef size_t (*CstlRawHashCodeCalcFunc)(const void *key, size_t keySize);

/**
 * @ingroup cstl_rawhash
 * 该函数把输入数据与key进行匹配比较。第一个入参是输入数据，第二个入参是对应的key。
 * 如果匹配，返回true，否则返回false。
 * @param key1     [IN] hash表中已保存的key。
 * @param key1Size [IN] hash表中key的空间大小，即为用户在CstlRawHashCreate时传入的keySize。
 * @param key2     [IN] 用户传入的待匹配的key。
 * @param key1Size [IN] 用户用户传入的待匹配的key的实际大小。
 */
typedef bool (*CstlRawHashMatchFunc)(const void *key1, size_t key1Size, const void *key2, size_t key2Size);

/**
 * @ingroup cstl_rawhash
 * 创建一个新的Hash表，返回Hash表的句柄
 * @param bktSize      [IN] hash桶的容量（最大个数）
 * @param keySize      [IN] key的最大长度
 * @param valueSize    [IN] value的最大长度（不含key长度）
 * @param hashCalcFunc [IN] hash值计算函数，如果为NULL，则使用默认函数
 * @param matchFunc    [IN] hash key匹配函数，如为NULL，则默认逐字节比较
 * @param keyFunc      [IN] 用户key的拷贝与释放函数。
 * @param valueFunc    [IN] 用户value的拷贝与释放函数。
 * @retval 非NULL  成功创建的hash表句柄。
 * @retval NULL    创建失败。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
CstlRawHash *CstlRawHashCreate(size_t bktSize,
                               size_t keySize,
                               size_t valueSize,
                               CstlRawHashCodeCalcFunc hashCalcFunc,
                               CstlRawHashMatchFunc matchFunc,
                               CstlRawDupFreeFuncPair *keyFunc,
                               CstlRawDupFreeFuncPair *valueFunc);

/**
 * @ingroup cstl_rawhash
 * @brief 插入hash数据
 * @par 描述：该接口用于插入<K, V>键值对
 * @param hashTbl       [IN] hash表的句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @param key           [IN] 指向待插入key的指针。
 * @param keySize       [IN] key的大小。
 * @param value         [IN] 指向待插入value的指针。
 * @param valueSize     [IN] value的大小。
 * @retval #CSTL_OK      插入成功
 * @retval #CSTL_ERROR   插入失败
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
int32_t CstlRawHashInsert(CstlRawHash *hashTbl, const void *key, size_t keySize, const void *value, size_t valueSize);

/**
 * @ingroup cstl_rawhash
 * @brief 查找节点
 * @par 描述: 根据key查找并返回节点value地址
 * @param hashTbl   [IN] hash表的句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @param key       [IN] key
 * @param keySize   [IN] key实际长度
 * @retval 非NULL hash数据地址（指向value的地址）
 * @retval NULL   无法查找到节点
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
void *CstlRawHashFind(const CstlRawHash *hashTbl, const void *key, size_t keySize);

/**
 * @ingroup cstl_rawhash
 * @brief 判断当前hash表是否为空
 * @par 描述: 该API用于判断当前hash表是否为空，为空返回true，否则返回false。
 * @param  hashTbl [IN] hash表句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @retval #true， 表示hash表为空
 * @retval #false，表示hash表非空。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
bool CstlRawHashEmpty(const CstlRawHash *hashTbl);

/**
 * @ingroup cstl_rawhash
 * @brief 获取hash表的节点数量
 * @par 描述: 用于获取hash表的节点数量，返回节点个数。
 * @param  hashTbl [IN] hash表句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @retval 返回的hash节点数。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
size_t CstlRawHashSize(const CstlRawHash *hashTbl);

/**
 * @ingroup cstl_rawhash
 * @brief 从hash表中移除指定结点。
 * @par 描述: 根据key查找到节点并删除，同时调用用户释放函数释放扩展资源。
 * @param hashTbl  [IN] hash表句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @param key      [IN] 移除节点key。
 * @param keySize  [IN] key实际长度
 * @retval  #CSTL_OK     删除成功。
 * @retval  #CSTL_ERROR  删除失败。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
int32_t CstlRawHashErase(CstlRawHash *hashTbl, const void *key, size_t keySize);

/**
 * @ingroup cstl_rawhash
 * @brief 删除hash表所有节点
 * @par 描述：删除所有节点，回收节点内存（hash表还在，只是没有成员）
 * @attention 如果用户数据中有扩展资源，则需要在创建时注册free钩子函数，这样可以先调该钩子释放用户数据中的资源。
 * @param  hashTbl [IN] hash表句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
void CstlRawHashClear(CstlRawHash *hashTbl);

/**
 * @ingroup cstl_rawhash
 * @brief 删除hash表
 * @par 描述：删除hash表，如里面有节点先删除节点，回收内存。
 * @attention 如果用户数据中有扩展资源，则需要在创建时注册free钩子函数，这样可以先调该钩子释放用户数据中的资源。
 * @param  hashTbl [IN] hash表句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
void CstlRawHashDestory(CstlRawHash *hashTbl);

/**
 * @ingroup cstl_rawhash
 * @brief 获取hash表中的第一个节点。
 * @par 描述：获取hash表中的第一个节点。
 * @param  hashTbl [IN]  hash表句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @param  key     [OUT] 指向第一个hash节点的key
 * @param  value   [OUT] 指向第一个hash节点的value
 * @retval #CSTL_OK      获取成功。
 * @retval #CSTL_ERROR   获取失败。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件.
 */
int32_t CstlRawHashFront(const CstlRawHash *hashTbl, void **key, void **value);

/**
 * @ingroup cstl_rawhash
 * @brief 获取hash表中的下一个节点。
 * @par 描述：获取hash表中的下一个节点。如根据key查找节点失败，则从根据key计算出来hash桶的下一个位置继续查找。
 * @param  hashTbl   [IN]  hash表句柄。取值范围为CstlRawHashCreate返回的合法指针。
 * @param  curkey    [IN]  当前hash节点的key
 * @param  keySize   [IN]  当前hash节点的key长度
 * @param  nextkey   [OUT] 下一个hash节点的key地址
 * @param  nextValue [OUT] 下一个hash节点的value地址
 * @retval #CSTL_OK      获取成功。
 * @retval #CSTL_ERROR   获取失败。
 * @par 依赖：无
 * @li cstl_rawhash.h：该接口声明所在的头文件。
 */
int32_t CstlRawHashNext(const CstlRawHash *hashTbl, const void *curkey, size_t keySize,
                        void **nextkey, void **nextValue);

#ifdef __cplusplus
}
#endif

#endif /*  CSTL_RAWHASH_H */