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
#ifndef NE_SOMEIP_BASE_NETYPEDEFINE_H
#define NE_SOMEIP_BASE_NETYPEDEFINE_H

# include "stdint.h"

/** ne_someip_base_compare_func
 * @data1: 
 * @data2: 
 *
 * 比较函数
 *
 * Returns: 0 data1和data2相等；> 0 data1大于data2；< 0 data1小于data2
 */
typedef int32_t (*ne_someip_base_compare_func)(const void* data1,const void* data2);
/** ne_someip_base_free_func
 * @data: 要释放的数据指针
 * @data2: 
 *
 * 释放数据
 *
 * Returns: 
 */
typedef void (*ne_someip_base_free_func)(void* data);

typedef int32_t (*ne_someip_base_hash_func)(void* key);

#endif