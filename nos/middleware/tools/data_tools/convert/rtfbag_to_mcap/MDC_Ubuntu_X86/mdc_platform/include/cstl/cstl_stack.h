/**
 * @file cstl_stack.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief cstl_stack 对外头文件
 * @details cstl_stack 对外头文件
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
 * @defgroup cstl_stack 栈
 * @ingroup cstl
 */
#ifndef CSTL_STACK_H
#define CSTL_STACK_H

#include <stdint.h>
#include <stdbool.h>
#include "cstl_vector.h"
#include "cstl_public.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup cstl_stack
 * cstl_stack控制块
 */
typedef struct CstlStackInfo CstlStack;

/**
 * @ingroup cstl_stack
 * @brief 创建堆栈。
 * @par 描述：创建一个堆栈并返回其控制块。
 * @attention 注意：堆栈支持动态扩展
 * @param  dataFunc    [IN]  用户资源拷贝、释放函数对
 * @retval 非NULL 指向堆栈控制块的指针。
 * @retval NULL   创建失败。
 * @par 依赖：无
 * @li cstl_stack.h：该接口声明所在的文件。
*/
CstlStack *CstlStackCreate(CstlDupFreeFuncPair *dataFunc);

/**
 * @ingroup cstl_stack
 * @brief 检查stack是否为空
 * @param stack   [IN] stack控制块
 * @retval #true  1，Stack为空。
 * @retval #false 0，Stack不为空。
 * @li cstl_stack.h：该接口声明所在的文件。
 */
bool CstlStackEmpty(const CstlStack *stack);

/**
 * @ingroup cstl_stack
 * @brief 获取stack中成员个数
 * @param stack  [IN] stack控制块
 * @retval stack成员个数
 * @li cstl_stack.h：该接口声明所在的文件。
 */
size_t CstlStackSize(const CstlStack *stack);

/**
 * @ingroup cstl_stack
 * @brief 本接口将输入的数据压入到指定堆栈，并移动栈顶指针到新的栈顶。
 * @par 描述：本接口将输入的数据压入到指定堆栈，并移动栈顶指针到新的栈顶。如果满，则push失败。
 * @attention \n
 * 1.当用户注册了CstlDupFreeFuncPair函数里面的dupFunc情况下，CstlStackPush调用需要传入size参数，作为dupFunc函数的入参。\n
 * 2.用户如果dupFunc函数注册为空，则size参数在函数内部逻辑将不被使用。
 * @param stack    [IN] stack控制块。
 * @param data     [IN] 用户数据。
 * @param dataSize [IN] 用户数据长度，若用户未注册dupFunc，该参数将不被使用
 * @retval #CSTL_OK    插入数据成功。
 * @retval #CSTL_ERROR 插入数据失败。
 * @par 依赖：无
 * @li cstl_stack.h：该接口声明所在的文件。
*/
int32_t CstlStackPush(CstlStack *stack, uintptr_t data, size_t dataSize);

/**
 * @ingroup cstl_stack
 * @brief 移除栈顶元素，并移动栈顶指针到新的栈顶。
 * @par 描述：移除栈顶元素，并移动栈顶指针到新的栈顶。如果空，则pop失败。
 * @param stack          [IN] stack控制块。
 * @retval #CSTL_OK       弹出数据成功。
 * @retval #CSTL_ERROR    弹出数据失败。
 * @par 依赖：无
 * @li cstl_stack.h：该接口声明所在的文件。
 */
int32_t CstlStackPop(CstlStack *stack);

/**
 * @ingroup cstl_stack
 * @brief 获取当前栈顶元素，而不移动栈顶指针。
 * @attention 注意：如果stack为空，则返回0。用户不能区分是空返回的0，还是真实数据就是0，因此需要提前识别stack空的场景。\n
 * 本接口仅用于获取栈顶元素后，但该元素礽位于栈中，请不要对其进行释放。
 * @param stack    [IN]    stack控制块。
 * @retval stack栈顶数据。
 * @par 依赖：无
 * @li cstl_stack.h：该接口声明所在的文件。
 */
uintptr_t CstlStackTop(const CstlStack *stack);

/**
 * @ingroup cstl_stack
 * @brief 删除堆栈。
 * @par 描述：删除堆栈并释放所有资源。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_stack.h：该接口声明所在的文件。
 */
void CstlStackDestroy(CstlStack *stack);

/**
 * @ingroup cstl_stack
 * @brief 清空堆栈。
 * @par 描述：清空并释放资源，但是保留栈控制块。
 * @param stack    [IN]    stack控制块。
 * @retval 无。
 * @par 依赖：无
 * @li cstl_stack.h：该接口声明所在的文件。
 */
void CstlStackClear(CstlStack *stack);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* CSTL_STACK_H */