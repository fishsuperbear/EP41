/**
 * @file cstl_map.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_map 对外头文件
 * @details 映射定义
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
 * @defgroup cstl_map 映射
 * @ingroup cstl
 */
#ifndef CSTL_MAP_H
#define CSTL_MAP_H

#include <stdbool.h>
#include <stddef.h>
#include "cstl_public.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @ingroup cstl_map
 * cstl_map控制块
 */
typedef struct CstlMapInfo CstlMap;

/**
 * @ingroup cstl_map
 * cstl_map迭代器
 */
typedef struct TagCstlMapNode *CstlMapIterator;

/**
 * @ingroup cstl_map
 * @brief 创建map。
 * @par 描述：创建一个map，并返回其控制块指针。
 * @attention \n
 * 1.map存储的key和value均为uintptr_t类型。\n
 * 2.用户未注册keyCmpFunc函数时，将使用默认比较函数，key会被默认为整型。\n
 * 3.当用户key或value为字符串或其他复杂类型时，应将其地址作为参数（需强转为uintptr_t类型），
     若该地址指向栈内存，建议注册keyFunc->dupFunc或keyFunc->dupFunc进行key或数据的拷贝，返回对应的堆内存地址。\n
 * 4.若用户key或value为字符串或其他复杂类型，释放map表或擦除节点时，
     用户需自行释放资源或注册keyFunc->freeFunc、valueFunc->freeFunc。
 * @param  keyCmpFunc   [IN]  key的比较函数。
 * @param  keyFunc      [IN]  key拷贝及释放函数对。
 * @param  valueFunc    [IN]  data拷贝及释放函数对。
 * @retval 指向map控制块的指针，NULL表示创建失败。
 * @par 依赖：无。
 * @li cstl_map.h：该接口声明所在的文件。
 */
CstlMap *CstlMapCreate(CstlKeyCmpFunc keyCmpFunc, CstlDupFreeFuncPair *keyFunc, CstlDupFreeFuncPair *valueFunc);

/**
 * @ingroup cstl_map
 * @brief 向map表中插入一个(K,V)对。
 * @par 描述：用户数据插入后，会调用钩子keyCmpFunc完成排序。
 * @attention \n
 * 1.不支持重复的key。\n
 * 2.key或value为int类型时，直接将值作为参数即可。\n
 * 3.当用户key或value为字符串或其他复杂类型时，应将其地址作为参数（需强转为uintptr_t类型），\n
     如char *key, 则入参应为(uintptr_t)key。
 * @param map           [IN] map控制块。
 * @param key           [IN] key或保存key的地址。
 * @param keySize       [IN] key拷贝长度，如未注册dupFunc，此参数将不使用
 * @param value         [IN] value或保存value的地址。
 * @param valueSize     [IN] value拷贝长度，如未注册dupFunc，此参数将不使用
 * @retval #CSTL_OK      插入成功。
 * @retval #CSTL_ERROR   插入失败。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的文件。
*/
int32_t CstlMapInsert(CstlMap *map, uintptr_t key, size_t keySize, uintptr_t value, size_t valueSize);

/**
 * @ingroup cstl_map
 * @brief 向map表中插入一个(K,V)对或更新K对应的V。
 * @par 描述：该接口用于插入新节点或更新节点。
 * @attention \n
 * 1.支持重复的key。\n
 * 2.当key不存在时，与CstlMapInsert接口保持一致。\n
 * 3.当key存在时，会更新key对应的value。\n
 * @param map           [IN] map控制块。
 * @param key           [IN] key或保存key的地址。
 * @param keySize       [IN] key拷贝长度，如未注册dupFunc，此参数将不使用
 * @param value         [IN] value或保存value的地址。
 * @param valueSize     [IN] value拷贝长度，如未注册dupFunc，此参数将不使用
 * @retval #CSTL_OK      插入或更新节点成功。
 * @retval #CSTL_ERROR   插入或更新节点失败。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的文件。
*/
int32_t CstlMapPut(CstlMap *map, uintptr_t key, size_t keySize, uintptr_t value, size_t valueSize);

/**
 * @ingroup cstl_map
 * @brief 删除一个数据，并释放节点内存。
 * @par 描述：删除map中key对应的数据，并释放节点内存。
 * @attention \n
 * 1.该接口用于从map表中删除key对应的节点。\n
 * 2.key对应的节点存在时，返回下一个节点的迭代器，
 *  若key不存在或对应节点为最后一个节点，则返回CstlMapIterEnd()。
 * @param map       [IN] map控制块。
 * @param key       [IN] 待删除的key。
 * @retval #被删除节点的下一个节点迭代器，如果被删除key不存在或为最后一个节点，则返回CstlMapIterEnd()。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的文件。
 * @see #CstlMapFind
 */
CstlMapIterator CstlMapErase(CstlMap *map, uintptr_t key);

/**
 * @ingroup cstl_map
 * @brief 获取map中key对应的数据。
 * @param map           [IN]  map控制块。
 * @param key           [IN]  用户指定的key。
 * @param value         [OUT] 保存key对应的数据。
 * @retval #CSTL_OK      获取成功。
 * @retval #CSTL_ERROR   获取失败，map为空指针、value为空指针或对应的key不存在。
 * @par 依赖：无。
 * @li cstl_map.h：该接口声明所在的文件。
 * @see #CstlMapFind
 */
int32_t CstlMapAt(const CstlMap *map, uintptr_t key, uintptr_t *value);

/**
 * @ingroup cstl_map
 * @brief 获取map中key对应的数据。
 * @param map       [IN] map控制块。
 * @param key       [IN] 待获取的key。
 * @retval #返回key对应的迭代器。如果key不存在，则返回CstlMapIterEnd()。
 * @par 依赖：无。
 * @li cstl_map.h：该接口声明所在的文件。
 */
CstlMapIterator CstlMapFind(const CstlMap *map, uintptr_t key);

/**
 * @ingroup cstl_map
 * @brief 获取map中成员个数
 * @param map  [IN] map控制块。
 * @retval #map成员个数
 * @retval #0，map为空指针。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的头文件。
 */
size_t CstlMapSize(const CstlMap *map);

/**
 * @ingroup cstl_map
 * @brief 判断map是否为空
 * @param map       [IN] map控制块。
 * @retval #true，  表示map为空或map为NULL指针
 * @retval #false， 表示map为非空。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的头文件。
 */
bool CstlMapEmpty(const CstlMap *map);

/**
 * @ingroup cstl_map
 * @brief 获取map表中的第一个节点的迭代器。
 * @param map  [IN] map控制块。
 * @retval #第一个迭代器。当map中无节点时，返回CstlMapIterEnd()。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的头文件。
 */
CstlMapIterator CstlMapIterBegin(const CstlMap *map);

/**
 * @ingroup cstl_map
 * @brief 获取map表中的下一个节点的迭代器。
 * @param map  [IN] map控制块。
 * @retval #下一个迭代器。当前迭代器已是最后一个节点时，返回#CstlMapIterEnd()。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的头文件。
 */
CstlMapIterator CstlMapIterNext(const CstlMap *map, CstlMapIterator it);

/**
 * @ingroup cstl_map
 * @brief 获取map表中最后一个节点之后预留的迭代器。
 * @param map  [IN] map控制块。
 * @retval 最后一个map节点之后预留的迭代器。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的头文件。
 */
CstlMapIterator CstlMapIterEnd(const CstlMap *map);

/**
 * @ingroup cstl_map
 * @brief 获取迭代器的key。
 * @par 描述：获取map表中迭代器当前key。
 * @attention \n
 *  1.当map为空指针或it等于#CstlMapIterEnd()时，接口返回0，该接口无法区分是错误码还是用户数据，
 *    用户调用该接口时必须保证map为合法指针，并且it不等于#CstlMapIterEnd()
 * @param map   [IN]  map控制块。
 * @param it    [IN]  当前迭代器。禁止传入非法指针。
 * @retval #迭代器对应的key。
 * @retval #0 it等于CstlMapIterEnd()
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的文件。
 */
uintptr_t CstlMapIterKey(const CstlMap *map, CstlMapIterator it);

/**
 * @ingroup cstl_map
 * @brief 获取迭代器的value。
 * @par 描述：获取map表中迭代器当前key。
 * @attention \n
 *  1.当map为空指针或it等于#CstlMapIterEnd()时，接口返回0，该接口无法区分是错误码还是用户数据，
 *    用户调用该接口时必须保证map为合法指针，并且it不等于#CstlMapIterEnd()
 * @param hashTbl  [IN]  map表的句柄。
 * @param it       [IN]  当前迭代器。禁止传入非法指针。
 * @retval #迭代器对应的value。
 * @retval #0 it等于CstlMapIterEnd()
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的文件。
 */
uintptr_t CstlMapIterValue(const CstlMap *map, CstlMapIterator it);

/**
 * @ingroup cstl_map
 * @brief 清空map中的数据
 * @par 描述：清空map中的数据，保留控制块。
 * @attention  \n
 *  1.该接口会删除map表中所有节点。\n
 *  2.若用户注册了key或data的资源释放函数，则会调用钩子进行用户资源释放。\n
 *  3.调用本接口后，句柄保留，可以继续使用。\n
 * @param  map [IN] map句柄。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的文件。
 */
void CstlMapClear(CstlMap *map);

/**
 * @ingroup cstl_map
 * @brief 删除map
 * @par 描述：删除map，如里面有成员先删除成员，再回收控制块内存。
 * @attention  \n
 *  1.该接口会删除map表中所有节点。\n
 *  2.若用户注册了key或data的资源释放函数，则会调用钩子进行用户资源释放。\n
 *  3.调用本接口后，map指针指向的内存被free，map表不可以再次访问。\n
 * @param  map [IN] map句柄。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_map.h：该接口声明所在的文件。
 */
void CstlMapDestory(CstlMap *map);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CSTL_MAP_H */