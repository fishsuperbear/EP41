/**
 * @file cstl_vector.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_vector 对外头文件
 * @details cstl_vector 向量对外头文件
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
 * @defgroup cstl_vector 内存向量管理
 * @ingroup cstl
 */

#ifndef CSTL_VECTOR_H
#define CSTL_VECTOR_H

#include <stdint.h>
#include <stddef.h>

#include "cstl_public.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cstl_vector
 * vector资源控制块
 */
typedef struct CstlVectorInfo CstlVector;

/**
 * @ingroup cstl_vector
 * @brief vector创建
 * @par 描述：创建一个vector实例
 * @attention
 * 创建一个新的vector，用户只需制定成员大小，默认成员容量为2（即可以容纳两个成员），2倍递增
 * @param  itemSize         [IN]  vector中单个元素的大小
 * @retval 返回vector控制块地址，创建成功
 * @retval NULL, 创建失败
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
CstlVector *CstlVectorCreate(size_t itemSize);

/**
 * @ingroup cstl_vector
 * @brief vector创建
 * @par 描述：创建一个vector实例
 * @attention
 * 创建一个新的vector，用户只需制定成员大小，默认成员容量为2（即可以容纳两个成员），2倍递增
 * @param  itemSize   [IN]  vector中单个元素的大小
 * @param  itemCap    [IN]  vector中初始成员容量
 * @param  delta      [IN]  vector成员伸缩变化率
 * @retval 返回vector控制块地址，创建成功
 * @retval NULL, 创建失败
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
CstlVector *CstlVectorRawCreate(size_t itemSize, size_t itemCap, uint32_t delta);

/**
 * @ingroup cstl_vector
 * @brief vector尾部添加元素
 * @par 描述：在vector末尾插入节点
 * @attention
 * 把data指针指向的内容拷贝到最后一个成员位置；如容量已满，则动态扩展（内存申请可能失败）后再拷贝。
 * @param  vector      [IN]  vector控制块
 * @param  data        [IN]  待插入节点数据（数据拷贝vector尾部）
 * @retval 0，插入成功
 * @retval 1，插入失败。
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
int32_t CstlVectorPushBack(CstlVector *vector, const void *data);

/**
 * @ingroup cstl_vector
 * @brief 访问指定节点
 * @par 描述：根据索引号，返回指定索引的节点指针
 * @attention 待访问元素的索引需小于Vector的Size
 * @param  vector        [IN]  vector控制块
 * @param  index         [IN]  待访问的节点索引
 * @retval 返回索引为index的节点指针
 * @retval NULL 查找失败
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
void *CstlVectorAt(const CstlVector *vector, size_t index);

/**
 * @ingroup cstl_vector
 * @brief 查询vector元素个数
 * @par 描述：查询一个vector中已经存储的元素的个数
 * @attention 无
 * @param  vector  [IN]  vector控制块
 * @retval  元素个数
 * @par 依赖：无
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
size_t CstlVectorSize(const CstlVector *vector);

/**
 * @ingroup cstl_vector
 * @brief vector删除元素
 * @par 描述：删除实例中的一个元素，后面的数据会往前搬移，不会导致数据空洞
 * @attention 待删除元素的索引要小于Vector的Size
 * @param  vector       [IN]  需要被删除元素的实例
 * @param  index        [IN]  待删除的元素所在的索引
 * @retval #CSTL_OK      删除成功
 * @retval #CSTL_ERROR   删除失败
 * @par 依赖：无
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
int32_t CstlVectorErase(CstlVector *vector, size_t index);

/**
 * @ingroup cstl_vector
 * @brief 清除vector所有元素
 * @par 描述：清除vector中的所有的元素，同时vector容量恢复到默认值。
 * @attention
 * 如果用户数据中有指针或其它资源，用户必须先行释放这些资源。
 * @param  vector       [IN]  待清除的vector实例
 * @param  freeFunc     [IN]  资源释放方法
 * @retval 无
 * @par 依赖：无
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
void CstlVectorClear(CstlVector *vector, CstlFreeFunc freeFunc);

/**
 * @ingroup cstl_vector
 * @brief 将vector中的元素进行排序
 * @par 描述：对vector中的元素进行排序
 * @attention 无
 * @param  vector             [IN]  被排序的vector
 * @param  compareFun         [IN]  排序方法
 * @retval 无
 * @par 依赖：无
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
void CstlVectorSort(CstlVector *vector, CstlDataCmpFunc cmpFunc);

/**
 * @ingroup cstl_vector
 * @brief 在vector内查找目标数据，返回其所在的指针
 * @par 描述：在vector内查找目标数据，返回其所在的指针
 * @attention 注意：在调本函数进行search前，用户必须先调 CstlVectorSort 进行排序，否则返回的结果没任何意义。
 * @param  vector      [IN]  被查找的vector
 * @param  data        [IN]  查找索引
 * @param  cmpFunc     [IN]  查找方法
 * @retval vector实例中目标元素所在的地址。如未找到，则返回NULL。
 * @par 依赖：无
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
void *CstlVectorSearch(const CstlVector *vector, const void *data, CstlDataCmpFunc cmpFunc);

/**
 * @ingroup cstl_vector
 * @brief vector销毁
 * @par 描述：销毁vector实例
 * @attention
 * 本接口不仅删除所有成员，还把控制块一并删除。如果用户数据中有指针或者其它资源，用户必须先释放这些资源。
 * @param  vector       [IN]  被销毁的实例
 * @param  freeFunc     [IN]  资源释放方法
 * @retval 无
 * @par 依赖：无
 * @li cstl_vector.h：该接口声明所在的头文件。
 */
void CstlVectorDestroy(CstlVector *vector, CstlFreeFunc freeFunc);

#ifdef __cplusplus
}
#endif

#endif /* CSTL_VECTOR_H */

