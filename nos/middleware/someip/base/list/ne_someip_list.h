/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/

#ifndef NE_SOMEIP_LIST_H
#define NE_SOMEIP_LIST_H

#ifdef  __cplusplus
extern "C" {
#endif

# include "ne_someip_base_netypedefine.h"
# include "stdbool.h"
# include "stdint.h"

typedef struct ne_someip_list ne_someip_list_t;
typedef struct ne_someip_list_element ne_someip_list_element_t;
typedef struct ne_someip_list_iterator ne_someip_list_iterator_t;

// define someip list element
typedef struct ne_someip_list_element
{
    void* data;                           /* element content that is defined by user */
    struct ne_someip_list_element* prev;  /* previous element pointer */
    struct ne_someip_list_element* next;  /* next element pointer */
} ne_someip_list_element_t;


/** ne_someip_list_foreach_func
 * @data: list的element数据
 * @user_data: 用户传入的透传参数
 *
 * list遍历函数
 *
 * Returns: TRUE 不需要继续遍历；FALSE 继续遍历list的下一element
 */
typedef bool (*ne_someip_list_foreach_func)(const void* data, void* user_data);

/** ne_someip_list_create
 * 创建一个新的list
 *
 * Returns: 新创建的list
 */
ne_someip_list_t* ne_someip_list_create();

/** ne_someip_list_destroy
 * @list:
 *
 * 删除list后的全部element，并释放list
 *
 * Returns:
 */
void ne_someip_list_destroy(ne_someip_list_t* list, ne_someip_base_free_func free_func);

/** ne_someip_list_append
 * @list:
 * @data: 要插入的数据
 *
 * 向list尾部插入一个element。如果list为空，插入的数据将成为一个新的list
 *
 * Returns: 插入后的list
 */
ne_someip_list_t* ne_someip_list_append(ne_someip_list_t* list, void* data);

/** ne_someip_list_prepend
 * @list:
 * @data: 要插入的数据
 *
 * 向list头部插入一个element。如果list为空，插入的数据将成为一个新的list
 *
 * Returns: 插入后的list
 */
ne_someip_list_t* ne_someip_list_prepend(ne_someip_list_t* list, void* data);

/** ne_someip_list_length
 * @list:
 *
 * 计算list的长度
 *
 * Returns: 插入后的list
 */
int32_t ne_someip_list_length(ne_someip_list_t* list);

/** ne_someip_list_remove_all
 * @list:
 *
 * 删除list后的全部element。
 *
 * Returns:
 */
void ne_someip_list_remove_all(ne_someip_list_t* list, ne_someip_base_free_func free_func);

/** ne_someip_list_remove_by_data
 * @list:
 * @data: 要删除的数据
 *
 * 从list中找到第一个数据为data的element，从list中移除
 * 如果指定了free_func，调用free_func释放data
 *
 * Returns: 删除后的list
 */
ne_someip_list_t* ne_someip_list_remove_by_data(ne_someip_list_t* list, void* data, ne_someip_base_free_func free_func);
/** ne_someip_list_remove_by_elem
 * @list:
 * @elem: 要删除的element
 *
 * 从list中移除element
 * 如果指定了free_func，调用free_func释放element的data
 *
 * Returns: 删除后的list
 */
ne_someip_list_t* ne_someip_list_remove_by_elem(ne_someip_list_t* list, ne_someip_list_element_t* elem, ne_someip_base_free_func free_func);
/** ne_someip_list_get_element
 * @list:
 * @n: 指定element序号(从0开始)
 *
 * 从list中获取第n个element，并返回
 *
 * Returns: 第n个element
 */
ne_someip_list_element_t* ne_someip_list_get_element(ne_someip_list_t* list, int32_t index);
/** ne_someip_list_get_data
 * @list:
 * @n: 指定element序号(从0开始)
 *
 * 从list中获取第n个element的data，并返回
 *
 * Returns: 第n个element的data
 */
void* ne_someip_list_get_data(ne_someip_list_t* list, int32_t index);
/** ne_someip_list_merge_list
 * @dest_list
 * @src_list
 * src list中的element拷贝一份（浅拷贝），追加到dest_list
 *
 * Returns: 合并后的list
 */
ne_someip_list_t* ne_someip_list_merge_list(ne_someip_list_t* dest_list, ne_someip_list_t* src_list);
/** ne_someip_list_merge_list
 * @first_list
 * @second_element
 * 从second element开始，将的element拷贝一份（浅拷贝），追加到first_list
 *
 * Returns: 合并后的list
 */
ne_someip_list_t* ne_someip_list_merge_element(ne_someip_list_t* first_list, ne_someip_list_element_t* second_element);
/** ne_someip_list_move_list
 * @dest_list
 * @src_list
 * 将src_list中的element移动到dest_list（浅拷贝）
 *
 * Returns: 合并后的list
 */
ne_someip_list_t* ne_someip_list_move_list(ne_someip_list_t* dest_list, ne_someip_list_t* src_list);
/** ne_someip_list_get_position
 * @elem: 指定element在list中的序号(从0开始)
 *
 * 获取指定element的在list中的序号
 *
 * Returns: element在list中的序号
 */
int32_t ne_someip_list_get_position(ne_someip_list_t* list, ne_someip_list_element_t* elem);
/** ne_someip_list_get_index
 * @list：
 * @data：
 *
 * 获取第一个包含data的element，在list中的序号(从0开始)
 *
 * Returns: element在list中的序号
 */
int32_t ne_someip_list_get_index(ne_someip_list_t* list, void* data);
/** ne_someip_list_first
 * @list：
 *
 * 返回list中的第一个element
 *
 * Returns: 成功 返回list中的第一个element的数据 失败 返回NULL
 */
ne_someip_list_element_t* ne_someip_list_first(ne_someip_list_t* list);
/** ne_someip_list_last
 * @list：
 *
 * 返回list中的最后一个element
 *
 * Returns: 成功 返回list中的最后一个element的数据 失败 返回NULL
 */
ne_someip_list_element_t* ne_someip_list_last(ne_someip_list_t* list);
/** ne_someip_list_insert
 * @list：
 * @data：要插入的数据
 * @pos：期望插入的位置
 *
 * 将data指入到list中的pos指定的位置。如果：
 * pos < 0，将插入到list尾
 * pos == 0，将插入到list头
 * pos超过list长度，将插入到list尾
 * list无效，插入的数据将成为一个新的list
 *
 * Returns: 成功 返回插入后的list 失败 返回NULL
 */
ne_someip_list_t* ne_someip_list_insert(ne_someip_list_t* list, void* data, int32_t pos);
/** ne_someip_list_insert_sorted
 * @list：
 * @data：要插入的数据
 * @cmp_func：排序比较函数
 * @user_data: 透传参数，提供给cmp_func比较函数使用。可以为NULL
 *
 * 将data指入到list中，在list中的pos，由cmp_func和list中既有element的data比较运算后确定。
 * list无效，插入的数据将成为一个新的list
 * data与list的element依次比较，直至找到cmp_func返回值大于0的element，data将插入到这个element之前
 * NEListInsertSorted只对当前data，通过排序函数计算插入位置，本函数不对既有list中的element进行排序
 * 如果要对list的所有element进行排序，参考NEListSorted
 *
 * Returns: 返回插入后的list
 */
ne_someip_list_t* ne_someip_list_insert_sorted(ne_someip_list_t* list, void* data, ne_someip_base_compare_func cmp_func, void* user_data);
/** ne_someip_list_sorted
 * @list：
 * @cmp_func：排序比较函数
 * @user_data: 透传参数，提供给cmp_func比较函数使用。可以为NULL
 *
 * 对list中的所有element排序，element在list中的pos，由cmp_func函数比较运算后确定。
 * cmp_func返回值小的element，排在list前部。
 *
 * Returns: 返回排序后的list
 */
ne_someip_list_t* ne_someip_list_sorted(ne_someip_list_t* list, ne_someip_base_compare_func cmp_func, void* user_data);
/** ne_someip_list_find
 * @list：
 * @data：要查找包含data数据的element
 * @cmp_func：比较函数，可以为NULL
 * @user_data: 透传参数，提供给cmp_func比较函数使用。可以为NULL
 *
 * 从list中找到第一个包含data数据的element。
 * 如果cmp_func为NULL，将直接比较data指针
 *
 * Returns: 成功 返回找到的element，失败返回NULL
 */
ne_someip_list_element_t* ne_someip_list_find(ne_someip_list_t* list, void* data, ne_someip_base_compare_func cmp_func, void* user_data);
/** ne_someip_list_foreach
 * @list：
 * @foreach_func：对每个element调用的函数, 参考ne_someip_list_foreach_func
 * @user_data: 透传参数，提供给foreach_func函数使用。可以为NULL
 *
 * 遍历整个list，从第一个element开始，对每个element调用foreach_func。
 *
 * Returns:
 */
void ne_someip_list_foreach(ne_someip_list_t* list, ne_someip_list_foreach_func foreach_func, void* user_data);


// ================== iterator ===================
// list.begin()
ne_someip_list_iterator_t* ne_someip_list_iterator_create(ne_someip_list_t* list);

void ne_someip_list_iterator_destroy(ne_someip_list_iterator_t* iterator);

// iter++
void ne_someip_list_iterator_next(ne_someip_list_iterator_t* iterator);

// iter + 5
void ne_someip_list_iterator_next_step(ne_someip_list_iterator_t* iterator, int32_t step);

// iter--
void ne_someip_list_iterator_prev(ne_someip_list_iterator_t* iterator);

// iter - 5
void ne_someip_list_iterator_prev_step(ne_someip_list_iterator_t* iterator, int32_t step);

// *iter, iter->XX
void* ne_someip_list_iterator_data(ne_someip_list_iterator_t* iterator);

// swap
void* ne_someip_list_iterator_swap(ne_someip_list_iterator_t* iterator1, ne_someip_list_iterator_t* iterator2);

void ne_someip_list_iterator_remove(ne_someip_list_iterator_t* iterator, ne_someip_base_free_func free_func);

bool ne_someip_list_iterator_valid(ne_someip_list_iterator_t* iterator);

#ifdef __cplusplus
}
#endif
#endif // NE_SOMEIP_LIST_H
/* EOF */