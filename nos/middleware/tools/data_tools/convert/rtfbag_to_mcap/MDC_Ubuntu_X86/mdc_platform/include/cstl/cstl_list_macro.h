/**
 * @file cstl_list_macro.h
 * @copyright Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * @brief 宏版本的链表
 * @details 从DOPRA下沉的宏链表
 * @date 2021-05-14
 * @version v0.1.0
 * *******************************************************************************************
 * @par 修改日志：
 * <table>
 * <tr><th>Date        <th>Version  <th>Description
 * <tr><td>2021-05-14  <td>0.1.0    <td>创建初始版本
 * </table>
 * *******************************************************************************************
 * @par 修改日志：
 * <table>
 * <tr><th>Date        <th>Version  <th>Description
 * <tr><td>2021-08-09  <td>1.0.0    <td>规范整改
 * </table>
 * *******************************************************************************************
 */

#ifndef CSTL_LIST_MACRO_H
#define CSTL_LIST_MACRO_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cpluscplus */

/**
 * @ingroup cstl_list_macro
 * 该结构被用来保存双向链表中节点的前向指针和后向指针。
 * 这个链表不包含实质的数据区，一般用于组织(串接)数据节点，参见 #CSTL_LIST_ENTRY 的注释
 */
struct TagCstlListHead {
    struct TagCstlListHead *next, *prev;
};

/**
 * @ingroup cstl_list_macro
 * 链表头。
 */
typedef struct TagCstlListHead CSTL_LIST_HEAD_S;

/**
 * @ingroup cstl_list_macro
 * 初始化链表(用户不应该直接用这个宏，而应该使用封装后的宏#CSTL_LIST_DECLARE_AND_INIT)。
 * 请参考#CSTL_LIST_DECLARE_AND_INIT说明
 * @param list [IN] 需要初始化的链表 (注意不要把链表地址传进来) (Note, not the address)
 */
#define CSTL_LIST_INIT_VAL(list_) \
    { \
        ((&(list_)), (&(list_))) \
    }

/**
 * @ingroup cstl_list_macro
 * 定义且初始化一个链表，把链表节点的前向指针与后向指针指向自己 (Declare a list and init it)
 * @param list_ [IN] 需要定义和初始化的链表变量名(注意不要把链表地址传进来)。
 */
#define CSTL_LIST_DECLARE_AND_INIT(list_) \
    CSTL_LIST_HEAD_S (list_) = CSTL_LIST_INIT_VAL(list_)

/**
 * @ingroup cstl_list_macro
 * 初始化链表(链表重用时的初始化)
 * @param head_ [IN] 链表头结点的地址(The address of the head of a list)
 */
#define CSTL_LIST_INIT(head_) \
    do { \
        (head_)->next = (head_); \
        (head_)->prev = (head_); \
    } while (false)

/**
 * @ingroup cstl_list_macro
 * 在链表指定位置后边插入节点 (Add an item to a list after a special location)
 * @param item_  [IN] 节点地址(The address of the item)
 * @param where_ [IN] 该节点插入位置的前一个节点地址。(The address where the item will be inserted after)
 */
#define CSTL_LIST_ADD(item_, where_) \
    do { \
        (item_)->next       = (where_)->next; \
        (item_)->prev       = (where_); \
        (where_)->next      = (item_); \
        (item_)->next->prev = (item_); \
    } while (false)

/**
 * @ingroup cstl_list_macro
 * 在链表指定位置前边插入节点 (Add an item to a list before a special location)
 * @param item_  [IN] 节点地址(The address of the item)
 * @param where_ [IN] 该节点插入位置的后一个节点地址。(The address where the item will be inserted before)
 */
#define CSTL_LIST_ADD_BEFORE(item_, where_) \
    CSTL_LIST_ADD((item_), (where_)->prev)

/**
 * @ingroup cstl_list_macro
 * 从链表中删除一个节点(Remove an item from a list)
 * @param item_ [IN] 待删除的节点(The address of the item to be removed)
 */
#define CSTL_LIST_REMOVE(item_) \
    do { \
        (item_)->prev->next = (item_)->next; \
        (item_)->next->prev = (item_)->prev; \
    } while (false)

/**
 * @ingroup cstl_list_macro
 * 检查链表是否为空(Judge whether a list is empty)
 * @param head_ [IN] 需要检查的链表(The address of the list to be judged)
 * @retval #true，链表为空。
 * @retval #false，链表不为空。
 */
#define CSTL_LIST_IS_EMPTY(head_) ((head_)->next == (head_))

/**
 * @ingroup cstl_list_macro
 * 遍历一个链表(Travel through a list)
 * @param head_ [IN] 需要遍历的链表(The head of a list )
 * @param item_ [IN] 遍历链表所用的缓存节点(A temporary list item for travelling the list)
 */
#define CSTL_LIST_FOR_EACH_ITEM(item_, head_) \
    for ((item_) = (head_)->next; (item_) != (head_); (item_) = (item_)->next)

/**
 * @ingroup cstl_list_macro
 * 安全遍历一个链表(Travel through a list safety)
 * @param head_ [IN] 需要遍历的链表(The head of a list)
 * @param temp_ [IN] 指向当前节点以便安全删除当前节点(pointer used to save current item so you can free item_ safety)
 * @param item_ [IN] 遍历链表所用的缓存节点(A temporary list item for travelling the list)
 */
#define CSTL_LIST_FOR_EACH_ITEM_SAFE(item_, temp_, head_) \
    for ((item_) = (head_)->next, (temp_) = (item_)->next; \
         (item_) != (head_); \
         (item_) = (temp_), (temp_) = (item_)->next)

/**
 * @ingroup cstl_list_macro
 * 倒序遍历一个链表(Traverse a list backwards)
 * @param head_ [IN] 待遍历的链表头节点地址(The head of a list)
 * @param item_ [IN] 遍历链表所用的缓存节点(The loop index variable)
 */
#define CSTL_LIST_FOR_EACH_ITEM_REV(item_, head_) \
    for ((item_) = (head_)->prev; (item_) != (head_); (item_) = (item_)->prev)

/**
 * @ingroup cstl_list_macro
 * 从指定节点开始倒序遍历一个链表(Traverse a list backwards from a certain node)
 * @param item_  [IN] 遍历链表所用的缓存节点(The loop index variable)
 * @param start_ [IN] 开始遍历的节点(The node to start traversing from)
 * @param head_  [IN] 待遍历的链表头节点地址(The head of a list)
 */
#define CSTL_LIST_FOR_EACH_ITEM_REV_FROM(item_, start_, head_) \
    for ((item_) = (start_); (item_) != (head_); (item_) = (item_)->prev)

/**
 * @ingroup cstl_list_macro
 * 通过链表某个节点(小结点)找到该节点所在结构(大节点)的起始地址(Find the entry of a struct through its member variable whose type_ is list item_)
 * @param item_   [IN] 特定节点变量(The address of a list item)
 * @param type_   [IN] 包含链表节点的大节点类型(The type of a struct which includes the list item)
 * @param member_ [IN] 结构体内的list节点成员名称(The member variable of the struct whose type is list item)
 * 说明:
 * 每个结构变量形成一个一个的大节点(包含数据和list小结点), 大节点是通过list 这个链表(小节点)串起来的。
 * ---------      ---------      ---------    --               ----
 * |  pre    |<---|  pre    |<---|  pre    |     |==>小结点          |
 * |  next   |--->|  next   |--->|  next   |     |                  |
 * ---------      ---------      ---------    --                    | ===> 大节点
 * |  data1  |    |  data1  |    |  data1  |                        |
 * |  data2  |    |  data2  |    |  data2  |                        |
 * ---------      ---------      ---------                     ----
 * 不直接用list作为大节点的原因是 list(CSTL_LIST_HEAD_S 类型)只有头尾指针，不包含数据区。
 * 这样 list链表可以适用挂接任意个数据的场合，具有通用性。
 */
#define CSTL_LIST_ENTRY(item_, type_, member_) \
    ((type_ *)((uint8_t *)(item_) - (uintptr_t)(&(((type_ *)0)->member_))))

#ifdef __cplusplus
}
#endif

#endif /* CSTL_LIST_MACRO_H */