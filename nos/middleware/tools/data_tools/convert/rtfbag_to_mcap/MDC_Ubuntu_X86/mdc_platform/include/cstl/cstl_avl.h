/**
 * @file cstl_avl.h
 * @Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @Description: AVL树 CstlAvl 对外头文件。
 * @Author: anonym
 * @date: 2021-05
 * Notes: 分离式AVL（一次性申请数据后，由AVL维护）
 *        节点中包含key和value，关于key和value：
 *        1、如果用户key、value是整形数据，且长度均 <= sizeof(uintptr_t) , 则直接保存key值和value值在节点内。
 *        2、如果用户key、value是其他类型，且长度 >= sizeof(uintptr_t) ，则key和value记录的是指针，
 *           指向真正的用户数据key或value，此时用户必须注册duplicate函数和free函数对。
 *        3、duplicate函数：用户需要先根据源数据长度申请内存，在把用户数据拷贝到申请的内存中去，最后返回分配的内存地址。
 *
 *        场景1：节点中存入的是key和value值                       场景2：节点中存入key指针和value
 *           key长度 <= sizeof(uintptr_t)                         key长度 >= sizeof(uintptr_t) 或
 *           value长度 <= sizeof(uintptr_t)                       value长度 <= sizeof(uintptr_t)
 *
 *                   CstlAvlNode                                         CstlAvlNode
 *               +----------------+                                +----------------+            +----------+
 *               |      .....     |                                |      .....     |            | userdata |
 *               +      .....     +                                +      .....     +            +----------+
 *               |      .....     |                                |      .....     |                 /|\
 *               +       key      +                                +      key    ----------------------+
 *               |      value     |                                |      value     |
 *               +----------------+                                +----------------+
 *
 *          场景3：节点中存入的是key和value指针                    场景4：节点中存入key指针和value指针
 *           key长度 <= sizeof(uintptr_t)                         key长度 >= sizeof(uintptr_t) 或
 *           value长度 >= sizeof(uintptr_t)                       value长度 >= sizeof(uintptr_t)
 *
 *                   CstlAvlNode                                         CstlAvlNode
 *               +----------------+                                +----------------+            +----------+
 *               |      .....     |                                |      .....     |            | userdata |
 *               +      .....     +                                +      .....     +            +----------+
 *               |      .....     |                                |      .....     |                 /|\
 *               +       key      +                                +      key    ----------------------+
 *               |      value----------------------+               |     value    ---------------------+
 *               +----------------+               \|/              +----------------+                 \|/
 *                                           +----------+                                        +----------+
 *                                           | userdata |                                        | userdata |
 *                                           +----------+                                        +----------+
 */

/**
 * @defgroup cstl_avl AVL树
 * @ingroup cstl
 */

#ifndef CSTL_AVL_H
#define CSTL_AVL_H

#include <stddef.h>
#include "cstl_public.h"

#ifdef __cplusplus
extern "C"{
#endif /* __cplusplus */

/**
 * @ingroup cstl_avl
 * AVL树句柄
 */
typedef struct TagAvlTree CstlAvlTree;

/**
 * @ingroup cstl_avl
 * @brief 创建一个新的AVL树，返回AVL树的句柄
 * @attention
 * 1、关于key和value的拷贝函数：\n
 * 如下场景不需要注册拷贝函数：如果是int型数据，且长度<=sizeof(uintptr_t)。\n
 * 其它场景必须注册拷贝函数：\n
 * a）是int型数据，但长度 >sizeof(uintptr_t)；\n
 * b）字符串；\n
 * c）用户自定义数据结构；\n
 * 2、关于key和value的free函数：如果注册了duplicate函数，就必须注册相应的free函数。\n
 * 3、keyFunc和valueFunc中的拷贝函数和释放函数针对的是用户数据内存；mallocFreeFunc中的内存申请函数和释放函数针对的是树和节点内存。\n
 * @par 如果nodeCap为零，则动态申请内存；如果不为0，则在创建时按照指定值一次性分配好内存
 * @param nodeCap        [IN] 树节点的最大数量
 * @param compareFunc    [IN] AVL树节点key比较函数，原型位于public头文件中，函数不得为NULL
 * @param keyFunc        [IN] key拷贝及释放函数对，如果用户未注册，则默认为key为整型
 * @param valueFunc      [IN] value拷贝及释放函数对，如果用户未注册，则默认为value为整型
 * @param mallocFreeFunc [IN] 树和节点内存申请内存释放函数对，如果用户未注册，则默认为申请函数为malloc,释放函数为free
 * @retval             非NULL  成功创建的AVL树句柄
 * @retval             NULL    创建失败或者参数非法
 * @li                 cstl_avl.h：该接口声明所在的头文件。
 */
CstlAvlTree *CstlAvlTreeCreate(size_t nodeCap,
                               CstlKeyCmpFunc compareFunc,
                               const CstlDupFreeFuncPair *keyFunc,
                               const CstlDupFreeFuncPair *valueFunc,
                               const CstlMallocFreeFuncPair *mallocFreeFunc);

/**
 * @ingroup cstl_avl
 * @brief 插入AVL节点数据
 * @par 描述：创建节点，根据key把其所在的节点插入AVL树中的合适位置。
 * @attention
 * 1.插入时会自行申请节点内存，key和value为保存用户key值和value值或指向key和value的地址。\n
 * 2.key值唯一不可重复。\n
 * 3.如果用户数据生命周期小于节点周期，则在创建树时注册拷贝函数和释放函数。\n
 * @param  tree      [IN] AVL树的句柄
 * @param  key       [IN] key或保存key的地址。
 * @param  keySize   [IN] key拷贝长度，若用户未注册key的dupFunc，此参数将不被使用
 * @param  value     [IN] value或保存value的地址。
 * @param  valueSize [IN] value拷贝长度，若用户未注册value的dupFunc，此参数将不被使用
 * @retval #CSTL_OK      插入成功
 * @retval #CSTL_ERROR   插入失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeInsert(CstlAvlTree *tree, uintptr_t key, size_t keySize, uintptr_t value, size_t valueSize);

/**
 * @ingroup cstl_avl
 * @brief 从AVL树中移除指定结点
 * @par 描述: 根据key查找到节点并删除（释放内存），同时释放节点内存
 * @param  tree     [IN] AVL树句柄
 * @param  key      [IN] 移除节点key
 * @retval #CSTL_OK     删除成功
 * @retval #CSTL_ERROR  删除失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeErase(CstlAvlTree *tree, uintptr_t key);

/**
 * @ingroup cstl_avl
 * @brief 根据key在AVL树上查找节点
 * @par 描述: 根据key在AVL树上查找节点，如果成功则返回CSTL_OK，出参为value，否则失败返回CSTL_ERROR
 * @param tree     [IN]  AVL树句柄
 * @param key      [IN]  key或保存key的地址
 * @param value    [OUT] 指向查找到节点的value的指针
 * @retval #CSTL_OK     查找成功
 * @retval #CSTL_ERROR  查找失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeFind(const CstlAvlTree *tree, uintptr_t key, uintptr_t *value);

/**
 * @ingroup cstl_avl
 * @brief 查找小于且最接近key或等于key值的节点
 * @par 描述: 根据key在AVL树上查找节点，如果成功则返回CSTL_OK，出参为value，否则失败返回CSTL_ERROR
 * @param tree     [IN]  AVL树句柄
 * @param key      [IN]  key或保存key的地址
 * @param value    [OUT] 指向查找到节点的value的指针
 * @retval #CSTL_OK     查找成功
 * @retval #CSTL_ERROR  查找失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeFindLessOrEqual(const CstlAvlTree *tree, uintptr_t key, uintptr_t *value);

/**
 * @ingroup cstl_avl
 * @brief 查找大于且最接近或等于给定key值的节点
 * @par 描述: 根据key在AVL树上查找节点，如果成功则返回CSTL_OK，出参为value，否则失败返回CSTL_ERROR
 * @param tree     [IN]  AVL树句柄
 * @param key      [IN]  key或保存key的地址
 * @param value    [OUT] 指向查找到节点的value的指针
 * @retval #CSTL_OK     查找成功
 * @retval #CSTL_ERROR  查找失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeFindBiggerOrEqual(const CstlAvlTree *tree, uintptr_t key, uintptr_t *value);

/**
 * @ingroup cstl_avl
 * @brief 获取AVL树中的第一个节点，最左叶子节点
 * @par 描述：获取AVL树中的第一个节点，最左叶子节点
 * @param  tree    [IN]  AVL树句柄
 * @param  key     [OUT] 指向最左叶节点key的指针
 * @param  value   [OUT] 指向最左叶节点value的指针
 * @retval #CSTL_OK     获取成功
 * @retval #CSTL_ERROR  获取失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeFront(const CstlAvlTree *tree, uintptr_t *key, uintptr_t *value);

/**
 * @ingroup cstl_avl
 * @brief 获取AVL树中的最后一个节点，最右叶子节点
 * @par 描述：获取AVL树中的最后一个节点，最右叶子节点
 * @param  tree    [IN]  AVL树句柄
 * @param  key     [OUT] 指向最右叶节点key的指针
 * @param  value   [OUT] 指向最右叶节点value的指针
 * @retval #CSTL_OK     获取成功
 * @retval #CSTL_ERROR  获取失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeBack(const CstlAvlTree *tree, uintptr_t *key, uintptr_t *value);

/**
 * @ingroup cstl_avl
 * @brief 查找AVL树中指定节点的前驱节点
 * @par 描述：查找AVL树中指定节点的前驱节点（比指定key小且最接近的节点）
 * @param  tree      [IN]  AVL树句柄
 * @param  key       [IN]  key或保存key的地址
 * @param  prevKey   [OUT] 指向前驱节点中key的指针
 * @param  prevValue [OUT] 指向前驱节点中value的指针
 * @retval #CSTL_OK     查找成功
 * @retval #CSTL_ERROR  查找失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodePrev(const CstlAvlTree *tree, uintptr_t key, uintptr_t *prevKey, uintptr_t *prevValue);

/**
 * @ingroup cstl_avl
 * @brief 查找AVL树中指定节点的后继节点（比指定key大且最接近的节点）
 * @par 描述：查找AVL树中指定节点的后继节点
 * @param  tree      [IN]  AVL树句柄
 * @param  key       [IN]  key或保存key的地址
 * @param  nextKey   [OUT] 指向后继节点中key的指针
 * @param  nextValue [OUT] 指向后继节点中value的指针
 * @retval #CSTL_OK     查找成功
 * @retval #CSTL_ERROR  查找失败或者参数非法
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeNext(const CstlAvlTree *tree, uintptr_t key, uintptr_t *nextKey, uintptr_t *nextValue);

/**
 * @ingroup cstl_avl
 * @brief 删除AVL树所有节点
 * @par 描述：删除所有节点，回收节点内存（AVL树还在，只是没有成员）。
 * @attention 如果用户数据中有资源，则需要在创建时注册free钩子函数，这样可以先调该钩子释放用户数据中的资源
 * @param  tree [IN]  AVL树句柄
 * @retval #CSTL_OK     清空成功
 * @retval #CSTL_ERROR  清空失败或者传入空指针
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlNodeClear(CstlAvlTree *tree);

/**
 * @ingroup cstl_avl
 * @brief 删除AVL树。
 * @par 描述：删除AVL树，先删除所有节点，最后释放句柄
 * @attention 如果用户数据中有资源，则需要在创建时注册free钩子函数，这样可以先调该钩子释放用户数据中的资源
 * @param  tree [IN] AVL树句柄
 * @retval #CSTL_OK     销毁成功
 * @retval #CSTL_ERROR  销毁失败或者传入空指针
 * @li cstl_avl.h：该接口声明所在的头文件。
 */
int32_t CstlAvlTreeDestroy(CstlAvlTree *tree);

#ifdef __cplusplus
}
#endif

#endif /* CSTL_AVL_H */