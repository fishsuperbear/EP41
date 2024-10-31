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
#ifndef NE_SOMEIP_OBJECT_H
#define NE_SOMEIP_OBJECT_H

#ifdef __cplusplus
extern "C" {
#endif

# include <pthread.h>
# include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define NEOBJECT_MEMBER \
    int32_t obj_ref_count; \
    pthread_mutex_t obj_ref_mutex;

/**
 * 引用计数本身并非线程安全
 * 执行typeName##Ref使用象引用计数递增时，必须保证此刻对象本身是有效的。
 * 对一个已经释放的对象执行typeName##Ref/typeName##Unref将导致未定义行为。
 */
#define NEOBJECT_FUNCTION(typeName) \
void typeName##_ref_count_init(typeName *obj) \
{ \
    pthread_mutex_init(&(obj->obj_ref_mutex),NULL); \
    obj->obj_ref_count = 1; \
} \
void typeName##_ref_count_deinit(typeName *obj) \
{ \
    obj->obj_ref_count = 0; \
    pthread_mutex_destroy(&(obj->obj_ref_mutex)); \
} \
typeName* typeName##_ref(typeName *obj) \
{ \
    pthread_mutex_lock(&(obj->obj_ref_mutex)); \
    if (0 >= obj->obj_ref_count) { \
        pthread_mutex_unlock(&(obj->obj_ref_mutex)); \
        return NULL; \
    } \
    ++obj->obj_ref_count; \
    pthread_mutex_unlock(&(obj->obj_ref_mutex)); \
    return obj; \
} \
void typeName##_unref(typeName *obj) \
{ \
    pthread_mutex_lock(&(obj->obj_ref_mutex)); \
    if ( 1 < obj->obj_ref_count) { \
        --obj->obj_ref_count; \
        pthread_mutex_unlock(&(obj->obj_ref_mutex)); \
    } else { \
        obj->obj_ref_count = 0; \
        pthread_mutex_unlock(&(obj->obj_ref_mutex)); \
        typeName##_free(obj); \
        obj = NULL; \
    } \
}

#ifdef __cplusplus
}
#endif
#endif // NE_SOMEIP_OBJECT_H
/* EOF */