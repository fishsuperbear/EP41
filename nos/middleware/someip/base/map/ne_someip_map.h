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
#ifndef NE_SOMEIP_MAP_H
#define NE_SOMEIP_MAP_H

#ifdef  __cplusplus
extern "C" {
#endif

# include "ne_someip_base_netypedefine.h"
# include "ne_someip_list.h"
# include "stdint.h"
# include "stdbool.h"


typedef struct ne_someip_map ne_someip_map_t;
typedef struct ne_someip_map_iter ne_someip_map_iter_t;
typedef uint32_t (*ne_someip_map_hash_func)(const void * key);



/** ne_someip_map_new
 * @hash_func: hash值生成函数。
 * @key_cmp_func: key的比较函数。ne_someip_map_t提供三个key比较函数：
 * @key_free_func: key-value对移除时，value的释放函数。可以为NULL
 * @value_free_func: key-value对移除时，key的释放函数。可以为NULL
 *
 * 创建ne_someip_map_t对象
 * 预定义了三个hash函数：ne_someip_map_tInt32HashFunc、ne_someip_map_tStringHashFunc、ne_someip_map_tPointerHashFunc
 * 预定义了三个key比较函数：ne_someip_map_tInt32CompareFunc、ne_someip_map_tStringCompareFunc、ne_someip_map_tPointerCompareFunc
 * 用户可以自己扩展hash函数和key比较函数
 *
 * Returns: ne_someip_map_t对象
 */
ne_someip_map_t* ne_someip_map_new(ne_someip_map_hash_func hash_func, ne_someip_base_compare_func key_cmp_func, ne_someip_base_free_func key_free_func, ne_someip_base_free_func value_free_func);
/** ne_someip_map_ref
 * @map:
 *
 * 递增ne_someip_map_t对象的引用计数，并返回ne_someip_map_t对象指针
 *
 * Returns: 返回ne_someip_map_t对象指针
 */
ne_someip_map_t* ne_someip_map_ref(ne_someip_map_t* map);
/** ne_someip_map_unref
 * @map:
 *
 * 递减ne_someip_map_t对象的引用计数
 * 引用计数减为0时，将释放对象
 *
 * Returns:
 */
void ne_someip_map_unref(ne_someip_map_t* map);
/** ne_someip_map_insert
 * @map:
 * @key:
 * @value:
 *
 * 把指定的key-value对插入到map
 * 如果map中已有相同的key，旧的key-value对将被移除（包括释放内存）
 * 注意！！！！：插入数据可能会导致map的大小改变，严禁遍历过程中插入数据
 *
 * Returns: true 成功 fasle 失败
 */
bool ne_someip_map_insert(ne_someip_map_t* map, void* key, void* value);
/** ne_someip_map_remove
 * @map:
 * @key:
 * @resize: 移除后,是否要重调整map的大小
 *
 * 从map中移除指定key对应的key-value对
 * 如果在遍历过程中，调用ne_someip_map_tRemove时，不可调整map大小，resize必须指定为FALSE。
 *
 * Returns: TRUE 成功；FALSE 失败
 */
bool ne_someip_map_remove(ne_someip_map_t* map, void* key, bool resize);
/** ne_someip_map_remove_all
 * @map:
 * @key:
 *
 * 从map中移除全部key和对应的value
 *
 * Returns: TRUE 成功；FALSE 失败
 */
void ne_someip_map_remove_all(ne_someip_map_t* map);
/** ne_someip_map_int32_hash_func
 * @key: Int32类型的key地址
 *
 *
 * 根据一个Int32的key转换出对应的hash值。
 *
 * Returns: hash值
 */
uint32_t ne_someip_map_int32_hash_func(const void* key);
/** ne_someip_map_int32_cmp_func
 * @k1: Int32类型的key地址
 * @k2: Int32类型的key地址
 *
 *
 * 比较两个Int32的key是否相等。
 *
 * Returns: TRUE 相等；FALSE 不相等
 */
int32_t ne_someip_map_int32_cmp_func(const void* k1, const void* k2);
/** ne_someip_map_string_hash_func
 * @key: string类型('\0'结尾的字符串)的key地址
 *
 *
 * 根据一个string类型的key转换出对应的hash值。
 *
 * Returns: hash值
 */
uint32_t ne_someip_map_string_hash_func(const void* key);
/** ne_someip_map_string_cmp_func
 * @k1: string类型('\0'结尾的字符串)的key地址 -> const char *
 * @k2: string类型('\0'结尾的字符串)的key地址 -> const char *
 *
 *
 * 比较两个string类型的key是否相等。
 *
 * Returns: TRUE 相等；FALSE 不相等
 */
int32_t ne_someip_map_string_cmp_func(const void* k1, const void* k2);
/** ne_someip_map_pointer_hash_func
 * @key: 指针类型的key
 *
 *
 * 根据一个指针类型的key转换出对应的hash值。
 *
 * Returns: hash值
 */
uint32_t ne_someip_map_pointer_hash_func(const void* key);
/** ne_someip_map_pointer_cmp_func
 * @k1: 指针类型的key
 * @k2: 指针类型的key
 *
 *
 * 比较两个指针类型的key是否相等。
 *
 * Returns: TRUE 相等；FALSE 不相等
 */
int32_t ne_someip_map_pointer_cmp_func(const void* k1, const void* k2);
/** ne_someip_map_find
 * @map:
 * @key:
 * @hash_return: 输出参数，key对应的hash值
 *
 * 从map中找到key对应的value
 *
 * Returns: 成功 返回key对应的value；失败 返回NULL
 */
void* ne_someip_map_find(ne_someip_map_t* map, const void* key, uint32_t* hash_return);
/** ne_someip_map_check
 * @map:
 * @key:
 *
 * 检查map中是否有对应的key
 *
 * Returns: TRUE 有；FALSE 没有
 */
bool ne_someip_map_check(ne_someip_map_t* map, void* key);
/** ne_someip_map_empty
 * @map:
 * @key:
 *
 * 检查map是否为空
 *
 * Returns: TRUE 空；FALSE 非空
 */
bool ne_someip_map_empty(ne_someip_map_t* map);
/** ne_someip_map_key_size
 * @map:
 * @key:
 *
 * 检查map中保存的key的个数
 *
 * Returns: 成功 对应map的key的个数 失败 -1
 */
int32_t ne_someip_map_key_size(ne_someip_map_t* map);
/** ne_someip_map_keys
 * @map:
 * @key:
 *
 * 返回当前map所有的key
 *
 * Returns: 成功 对应map的key的list(浅拷贝) 失败 NULL
 */
ne_someip_list_t* ne_someip_map_keys(ne_someip_map_t* map);
/** ne_someip_map_values
 * @map:
 * @key:
 *
 * 返回当前map所有的values
 *
 * Returns: 成功 对应map的values的list(浅拷贝) 失败 NULL
 */
ne_someip_list_t* ne_someip_map_values(ne_someip_map_t* map);
/** ne_someip_map_iter_new
 * @map:
 *
 * 创建遍历map的迭代器
 *
 * Returns: map的迭代器
 */
ne_someip_map_iter_t* ne_someip_map_iter_new(ne_someip_map_t* map);
/** ne_someip_map_iter_destroy
 * @iter:
 *
 * 销毁map的迭代器
 *
 * Returns: map的迭代器
 */
void ne_someip_map_iter_destroy(ne_someip_map_iter_t* iter);
/** ne_someip_map_iter_next
 * @iter:
 * @key: 输出参数，迭代器指向key-value对的key
 * @value:输出参数，迭代器指向key-value对的value
 *
 * 迭代器迁移至下一key-value对，并取回key和value
 *
 * Returns: TRUE 成功；FALSE 迭代结束
 */
bool ne_someip_map_iter_next(ne_someip_map_iter_t* iter, void** key, void** value);

#ifdef __cplusplus
}
#endif
#endif