/**
 * @file cstl_rawavl.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief AVL树头文件
 * @details \n
 *    1、节点、key和value在一起，由用户调 CstlRawAvlNodeAlloc 申请，然后用户自行构建key和value；\n
 *       对于key和value无特殊限制，只要不超过AVL树创建时指定的keySize和valueSize即可。\n
 *       注意：如果key是字符串，要考虑到最后'\0'的位置。\n
 *    2、如果用户数据中有指针，则必须在创建AVL树时注册freeFunc函数，即下图中场景2的其它数据需用户自行释放，\n
 *       否则会有内存泄露的风险。\n
 *    \n
 *    场景1：key、value中没有二级指针       场景2：key、value中有二级指针 \n
 *                                                                       +--------+ \n
 *      +----------------+                  +----------------+           | others | \n
 *      | CstlRawAvlNode |                  | CstlRawAvlNode |           +--------+ \n
 *      +----------------+                  +----------------+              /|\     \n
 *      |       key      |                  |       key      |---------------+      \n
 *      +----------------+                  +----------------+                      \n
 *      |      value     |                  |      value     |---------------+      \n
 *      +----------------+                  +----------------+              \|/     \n
 *                                                                       +--------+ \n
 *                                                                       | others | \n
 *                                                                       +--------+
 * @date 2021-04-15
 * @version v1.0.0
 * *******************************************************************************************
 * @par 修改日志：
 * <table>
 * <tr><th>Date        <th>Version  <th>Description
 * <tr><td>2021-04-15  <td>1.0.0    <td>创建初始版本
 * </table>
 * *******************************************************************************************
 */

/**
 * @defgroup cstl_rawavl AVL树
 * @ingroup cstl
 */
#ifndef CSTL_RAWAVL_H
#define CSTL_RAWAVL_H

#include <stddef.h>
#include "cstl_public.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cstl_rawavl
 * cstl_rawavl树句柄
 */
typedef struct TagRawAvlTree CstlRawAvlTree;

/**
 * @ingroup cstl_rawavl
 * @brief AVL树节点key值比较函数原型。
 * @par 描述：AVL树节点key值比较函数原型，用于排序和查询场景。
 * @attention 注意：这里只定义了比较函数原型，由于不知道数据类型和长度，因此钩子函数需要业务自己实现。默认逐字节比较。
 *            <key1> - <key2> 升序排序
 *            <key2> - <key1> 降序排序
 * @param key1     [IN] key1指针
 * @param key1Size [IN] key1长度
 * @param key2     [IN] key2指针
 * @param key2Size [IN] key2长度
 * @retval >0 key1较大
 * @retval =0 两者相等
 * @retval <0 key2较大
 */
typedef int32_t (*CstlRawAvlCompareFunc)(const void *key1, size_t key1Size, const void *key2, size_t key2Size);

/**
 * @ingroup cstl_rawavl
 * @brief 用户数据内存释放函数原型。
 * @par 描述：资源释放函数原型。用户调erase函数删除节点时，用户数据中可能含有私有资源，这时需要用户显式提供释放函数。
 * @param key    [IN] 指向节点中用户key空间的指针
 * @param value  [IN] 指向节点中用户value空间的指针
 * @retval 无
 */
typedef void (*CstlRawAvlFreeFunc)(void *key, void *value);

/**
 * @ingroup cstl_rawavl
 * @brief 创建一个新的AVL树，返回AVL树的句柄。
 * @par 描述：创建一个新的AVL树，返回AVL树的句柄。\n
 * 如果nodeCap为零，则动态申请内存；如果不为0，则在创建时按照指定值一次性分配好内存。
 * @param keySize      [IN] key的最大长度
 * @param valueSize    [IN] value的最大长度（不含key长度）
 * @param nodeCap      [IN] 树节点的最大数量
 * @param compareFunc  [IN] AVL树节点key比较函数，函数不得为NULL
 * @param freeFunc     [IN] 用户数据资源释放函数。如树节点的key、value中不含有私有资源（指针、句柄等），则本参数赋NULL即可
 * @retval 非NULL  成功创建的AVL树句柄
 * @retval NULL    创建失败
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
CstlRawAvlTree *CstlRawAvlTreeCreate(size_t keySize,
                                     size_t valueSize,
                                     size_t nodeCap,
                                     CstlRawAvlCompareFunc compareFunc,
                                     CstlRawAvlFreeFunc freeFunc);

/**
 * @ingroup cstl_rawavl
 * @brief 为AVL树节点分配内存。
 * @par 描述：根据AVL树创建时指定的大小分配内存（基础节点数据大小 + keySize + valueSize）。\n
 * @attention \n
 * 1.调用接口CstlRawAvlNodeInsert向AVL树中插入节点前，必须先调用本接口为插入节点的数据元素分配内存，并向key和value进行数据填充。
 * @param  tree     [IN]  AVL树句柄
 * @param  key      [OUT] 分配的key地址
 * @param  value    [OUT] 分配的value地址
 * @retval #CSTL_OK     内存分配成功
 * @retval #CSTL_ERROR  内存分配失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodeAlloc(CstlRawAvlTree *tree, void **key, void **value);

/**
 * @ingroup cstl_rawavl
 * @brief 插入AVL节点数据。
 * @par 描述：根据key地址把其所在的节点插入AVL树中的合适位置。
 * @attention \n
 * 1.该接口入参key和value必须来自于CstlRawAvlNodeAlloc接口的出参，且调用之前需向key和value进行数据填充。\n
 * 2.CstlRawAvlNodeInsert与CstlRawAvlNodeAlloc接口搭配使用，且key值唯一不可重复。\n
 * 3.若CstlRawAvlNodeInsert失败，其key地址所在的树节点内存将被释放，但不调用用户注册的freeFunc钩子释放用户的二级内存。\n
 * 4.若CstlRawAvlNodeInsert失败，需重新先调用CstlRawAvlNodeAlloc为树节点分配内存再进行插入操作。 \n
 * 5.用户key或者value指针中的内存当插入失败后需要用户自行释放
 * @param  tree      [IN] AVL树的句柄
 * @param  key       [IN] key地址（CstlRawAvlNodeAlloc返回的key地址）
 * @param  value     [IN] value地址（CstlRawAvlNodeAlloc返回的value地址）
 * @retval #CSTL_OK      插入成功
 * @retval #CSTL_ERROR   插入失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodeInsert(CstlRawAvlTree *tree, const void *key, const void *value);

/**
 * @ingroup cstl_rawavl
 * @brief 从AVL树中移除指定结点。
 * @par 描述: 根据key查找到节点并删除（释放内存）。
 * @param  tree     [IN] AVL树句柄
 * @param  key      [IN] key指针
 * @param  keySize  [IN] key长度
 * @retval #CSTL_OK     删除成功
 * @retval #CSTL_ERROR  删除失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodeErase(CstlRawAvlTree *tree, const void *key, size_t keySize);

/**
 * @ingroup cstl_rawavl
 * @brief 根据key在AVL树上查找节点。
 * @par 描述: 根据key在AVL树上查找节点。如果存在，则返回节点的value地址；否则，返回NULL。
 * @attention 返回值是指向value的地址，而不是节点地址。
 * @param tree     [IN]  AVL树句柄
 * @param key      [IN]  key指针
 * @param keySize  [IN]  key长度
 * @retval 非NULL  查找成功，返回指向这个节点value的地址
 * @retval NULL    查找失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
void *CstlRawAvlNodeFind(const CstlRawAvlTree *tree, const void *key, size_t keySize);

/**
 * @ingroup cstl_rawavl
 * @brief 查找小于且最接近key或等于key值的节点。
 * @par 描述: 查找小于或等于给定key值的节点，如果存在，则返回节点的value地址；否则，返回NULL。
 * @attention 返回值是指向value的地址，而不是节点地址。
 * @param tree     [IN]  AVL树句柄
 * @param key      [IN]  key指针
 * @param keySize  [IN]  key长度
 * @retval 非NULL  查找成功，返回指向这个节点的value地址
 * @retval NULL    查找失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
void *CstlRawAvlNodeFindLessOrEqual(const CstlRawAvlTree *tree, const void *key, size_t keySize);

/**
 * @ingroup cstl_rawavl
 * @brief 查找大于且最接近或等于给定key值的节点。
 * @par 描述: 查找大于或等于给定key值的节点，如果存在，则返回节点的value地址；否则，返回NULL。
 * @attention 返回值是指向value的地址，而不是节点地址。
 * @param tree     [IN]  AVL树句柄
 * @param key      [IN]  key指针
 * @param keySize  [IN]  key长度
 * @retval 非NULL  查找成功，返回指向这个节点的value地址
 * @retval NULL    查找失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
void *CstlRawAvlNodeFindBiggerOrEqual(const CstlRawAvlTree *tree, const void *key, size_t keySize);

/**
 * @ingroup cstl_rawavl
 * @brief 获取AVL树中的第一个节点，最左叶子节点。
 * @par 描述：获取AVL树中的第一个节点，最左叶子节点。
 * @param  tree    [IN]  AVL树句柄
 * @param  key     [OUT] 指向最左叶节点key的指针
 * @param  value   [OUT] 指向最左叶节点value的指针
 * @retval #CSTL_OK     获取成功
 * @retval #CSTL_ERROR  获取失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodeFront(const CstlRawAvlTree *tree, void **key, void **value);

/**
 * @ingroup cstl_rawavl
 * @brief 获取AVL树中的最后一个节点，最右叶子节点。
 * @par 描述：获取AVL树中的最后一个节点，最右叶子节点。
 * @param  tree    [IN]  AVL树句柄
 * @param  key     [OUT] 指向最右叶节点key的指针
 * @param  value   [OUT] 指向最右叶节点value的指针
 * @retval #CSTL_OK     获取成功
 * @retval #CSTL_ERROR  获取失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodeBack(const CstlRawAvlTree *tree, void **key, void **value);

/**
 * @ingroup cstl_rawavl
 * @brief 查找AVL树中指定节点的前驱节点。
 * @par 描述：查找AVL树中指定节点的前驱节点（比指定key小且最接近的节点）。
 * @param  tree      [IN]  AVL树句柄
 * @param  key       [IN]  key指针
 * @param  keySize   [IN]  key长度
 * @param  prevKey   [OUT] 指向前驱节点中key的指针
 * @param  prevValue [OUT] 指向前驱节点中value的指针
 * @retval #CSTL_OK     查找成功
 * @retval #CSTL_ERROR  查找失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodePrev(
    const CstlRawAvlTree *tree, const void *key, size_t keySize, void **prevKey, void **prevValue);

/**
 * @ingroup cstl_rawavl
 * @brief 查找AVL树中指定节点的后继节点。
 * @par 描述：查找AVL树中指定节点的后继节点（比指定key大且最接近的节点）。
 * @param  tree      [IN]  AVL树句柄
 * @param  key       [IN]  key指针
 * @param  keySize   [IN]  key长度
 * @param  nextKey   [OUT] 指向后继节点中key的指针
 * @param  nextValue [OUT] 指向后继节点中value的指针
 * @retval #CSTL_OK     查找成功
 * @retval #CSTL_ERROR  查找失败或者参数非法
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodeNext(
    const CstlRawAvlTree *tree, const void *key, size_t keySize, void **nextKey, void **nextValue);

/**
 * @ingroup cstl_rawavl
 * @brief 删除AVL树所有节点。
 * @par 描述：删除所有节点，回收节点内存（AVL树还在，只是没有成员）。
 * @attention 如果用户数据中有资源，则需要在创建时注册free钩子函数，这样可以先调该钩子释放用户数据中的资源。
 * @param  tree [IN]  AVL树句柄
 * @retval #CSTL_OK     清空成功
 * @retval #CSTL_ERROR  清空失败或者传入空指针
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlNodeClear(CstlRawAvlTree *tree);

/**
 * @ingroup cstl_rawavl
 * @brief 销毁AVL树。
 * @par 描述：销毁AVL树，先删除所有节点，最后释放句柄。
 * @attention 如果用户数据中有资源，则需要在创建时注册free钩子函数，这样可以先调该钩子释放用户数据中的资源。
 * @param  tree [IN] AVL树句柄
 * @retval #CSTL_OK     销毁成功
 * @retval #CSTL_ERROR  销毁失败或者传入空指针
 * @li cstl_rawavl.h：该接口声明所在的头文件。
 */
int32_t CstlRawAvlTreeDestroy(CstlRawAvlTree *tree);

#ifdef __cplusplus
}
#endif

#endif /* CSTL_RAWAVL_H */
