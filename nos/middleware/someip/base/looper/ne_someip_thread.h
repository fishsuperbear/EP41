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
#ifndef NE_SOMEIP_THREAD_H
#define NE_SOMEIP_THREAD_H

#ifdef  __cplusplus
extern "C" {
#endif

# include <stdbool.h>
# include <stdint.h>
# include "ne_someip_looper.h"


typedef struct ne_someip_thread ne_someip_thread_t;
typedef void (*ne_someip_thread_user_main_func) (void *user_data);
typedef void (*ne_someip_thread_user_free_func) (void *user_data);

void ne_someip_thread_t_free(ne_someip_thread_t* thread);

/** ne_someip_thread_new
 * @name: ne_someip_thread_t 线程名
 * @thread_func: 线程主函数
 * @free_func: ne_someip_thread对象销毁时，调用free_func释放用户资源
 * @user_data: 线程主函数输入参数
 *
 * 创建ne_someip_thread对象，ne_someip_thread对象初始引用计数为1
 *
 * Returns: 返回ne_someip_thread对象指针
 */
ne_someip_thread_t* ne_someip_thread_new(char* name, ne_someip_thread_user_main_func thread_func, ne_someip_thread_user_free_func free_func, void* user_data);

/** ne_someip_thread_new_looper
 * @name: ne_someip_thread_t 线程名
 * @thread_func: 线程主函数:默认为ne_someip_looper_run，也可用户自己指定
 * @free_func: ne_someip_thread对象销毁时，调用free_func释放用户资源，没有特殊要求填NULL
 * @user_data: 线程主函数输入参数
 *
 * 创建ne_someip_thread对象，ne_someip_thread对象初始引用计数为1，并为传入的looper对象+1
 * 注意：使用本函数创建的thread对象，在释放的时候（ne_someip_thread_t_free）会负责对传入的looper对象-1
 *
 * Returns: 返回ne_someip_thread对象指针
 */
ne_someip_thread_t* ne_someip_thread_new_looper(char* name, ne_someip_thread_user_main_func thread_func, ne_someip_thread_user_free_func free_func, ne_someip_looper_t* looper);

/** ne_someip_thread_ref
 * @thread:
 *
 * 增加ne_someip_thread对象的引用计数
 *
 * Returns: 返回ne_someip_thread对象指针
 */
ne_someip_thread_t* ne_someip_thread_ref(ne_someip_thread_t *thread);

/** ne_someip_thread_unref
 * @thread:
 *
 * 减少ne_someip_thread对象的引用计数
 *
 * Returns:
 */
void ne_someip_thread_unref(ne_someip_thread_t *thread);

/** ne_someip_thread_is_running
 * @thread:
 *
 * 获取ne_someip_thread的运行状态
 *
 * Returns: 0:未Start或已经Stop, 1: 已经Start，运行中
 */
bool ne_someip_thread_is_running(ne_someip_thread_t *thread);

/** ne_someip_thread_set_name
 * @thread:
 * @name:
 *
 * 设置线程名
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_thread_set_name(ne_someip_thread_t *thread, char* name);

/** ne_someip_thread_get_name
 * @thread:
 *
 * 获取线程名
 *
 * Returns: 成功返回线程名，失败返回NULL
 */
char* ne_someip_thread_get_name(ne_someip_thread_t *thread);

/** ne_someip_thread_start
 * @thread:
 *
 * 启动线程
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_thread_start(ne_someip_thread_t *thread);

/** ne_someip_thread_stop
 * @thread:
 *
 * 停止线程
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_thread_stop(ne_someip_thread_t *thread);

/** ne_someip_thread_self
 *
 * 获取当前线程ne_someip_thread对象
 * 在非ne_someip_thread创建的线程上调用本函数，将失败返回NULL
 *
 * Returns: 成功返回ne_someip_thread对象指针，失败返回NULL
 */
ne_someip_thread_t* ne_someip_thread_self();

/** ne_someip_thread_wait
 *
 * 当前运行在thread线程中，调用该函数进入挂起状态
 *
 * Returns: 0: 成功, -1: 失败
 */
int32_t ne_someip_thread_wait(ne_someip_thread_t* thread);

/** ne_someip_thread_notify
 *
 * 唤醒对应线程
 *
 * Returns: 0: 成功, -1: 失败
 */
int32_t ne_someip_thread_notify(ne_someip_thread_t* thread);

/** ne_someip_thread_get_tid
 * 
 * 获取线程id
 * 
 * Return: id
 */
pthread_t ne_someip_thread_get_tid(ne_someip_thread_t* thread);


#ifdef  __cplusplus
}
#endif
#endif // NE_SOMEIP_THREAD_H
/* EOF */
