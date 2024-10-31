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

#ifndef NE_SOMEIP_LOOPER_H
#define NE_SOMEIP_LOOPER_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <pthread.h>

/* ne_someip_looper_t */
typedef struct ne_someip_looper ne_someip_looper_t;
typedef struct {
    int32_t runnable_max_cnt;
    int32_t runnable_min_cnt;
} ne_someip_looper_config_t;
typedef struct {
    void (*run) (void *user_data);
    void (*free) (void *user_data);
    void *user_data;
} ne_someip_looper_runnable_t;

/* ne_someip_looper_timer_t */
typedef struct ne_someip_looper_timer ne_someip_looper_timer_t;
typedef struct {
    void (*run) (void *user_data);
    void (*free) (void *user_data);
    void *user_data;
} ne_someip_looper_timer_runnable_t;
typedef enum {
    NE_LOOPER_TIMER_TYPE_ABSOLUTE,				// 单次定时器，时间参数为绝对时间
    NE_LOOPER_TIMER_TYPE_INTERVAL_ONE_SHOT,		// 单次定时器，时间参数为相对时间
    NE_LOOPER_TIMER_TYPE_INTERVAL_CYCLE,		// 循环定时器，时间参数为间隔时间
    NE_LOOPER_TIMER_TYPE_MAX
} ne_someip_looper_timer_type_t;
typedef enum {
    NE_LOOPER_TIMER_STATE_STOP,					// 刚创建或调用Stop
    NE_LOOPER_TIMER_STATE_RUNNING,				// 定时器开启，运行中
    NE_LOOPER_TIMER_STATE_IDLE,					// 定时器到时，且非循环计时
    NE_LOOPER_TIMER_STATE_MAX
} ne_someip_looper_timer_state_t;

/* ne_someip_looper_io_source_t */
typedef struct ne_someip_looper_io_source ne_someip_looper_io_source_t;
typedef enum {
    NE_LOOPER_IO_SOURCE_CONDITION_READ = 0x01,
    NE_LOOPER_IO_SOURCE_CONDITION_WRITE = 0x02,
    NE_LOOPER_IO_SOURCE_CONDITION_ERROR = 0x04,
    NE_LOOPER_IO_SOURCE_CONDITION_MAX
} ne_someip_looper_io_source_condition_t;
typedef struct {
    void (*run) (ne_someip_looper_io_source_t* self, void *user_data, int32_t condition);
    void (*free) (void *user_data);
    void *user_data;
} ne_someip_looper_io_source_runnable_t;
typedef enum {
    NE_LOOPER_IO_SOURCE_STATE_STOP,
    NE_LOOPER_IO_SOURCE_STATE_STARTED,
    NE_LOOPER_IO_SOURCE_STATE_MAX
} ne_someip_looper_io_source_state_t;

void ne_someip_looper_t_free(void* data);
void ne_someip_looper_timer_t_free(void* data);
void ne_someip_looper_io_source_t_free(void* data);

/* ne_someip_looper_t */
/** ne_someip_looper_new
 * @config:
 *
 * 创建ne_someip_looper对象，ne_someip_looper对象初始引用计数为1
 *
 * Returns: 返回ne_someip_looper对象指针
 */
ne_someip_looper_t* ne_someip_looper_new(ne_someip_looper_config_t* config);

/** ne_someip_looper_ref
 * @looper:
 *
 * 增加ne_someip_looper对象的引用计数
 *
 * Returns: 返回ne_someip_looper对象指针
 */
ne_someip_looper_t* ne_someip_looper_ref(ne_someip_looper_t* looper);
/** ne_someip_looper_unref
 * @looper:
 *
 * 减少ne_someip_looper对象的引用计数
 *
 * Returns: void
 */
void ne_someip_looper_unref(ne_someip_looper_t* looper);
/** ne_someip_looper_run
 * @looper:
 *
 * ne_someip_looper主函数，一直运行直至ne_someip_looper_quit被成功调用
 *
 * Returns:
 */
void ne_someip_looper_run(void* temp_looper);
/** ne_someip_looper_quit
 * @looper:
 *
 * 通知ne_someip_looper主函数退出
 *
 * Returns: 0:成功, -1: 唤醒失败
 */
int32_t ne_someip_looper_quit(ne_someip_looper_t* looper);
/** ne_someip_looper_runnable_task_create_and_post
 * @looper:
 * @run_func:
 * @free_func:
 * @user_data:
 *
 * 创建一个ne_someip_looper_runnable_t，并post到对应的looper中
 *
 * Returns: 成功:0，失败:其他
 */

int32_t ne_someip_looper_runnable_task_create_and_post(ne_someip_looper_t* looper, void (*run_func) (void *user_data), void (*free_func) (void *user_data), void* user_data);
/** ne_someip_looper_runnable_task_create
 * @run_func:
 * @free_func:
 * @user_data:
 *
 * 创建一个ne_someip_looper_runnable_t，用于post
 * 并赋值run，free，user_data
 *
 * Returns: 成功:创建成功的runnable，失败:NULL
 */
ne_someip_looper_runnable_t* ne_someip_looper_runnable_task_create(void (*run_func) (void *user_data), void (*free_func) (void *user_data), void* user_data);
/** ne_someip_looper_runable_task_destory
 * @run_func:
 * @free_func:
 * @user_data:
 *
 * 释放ne_someip_looper_runnable_t
 *
 * Returns:
 */
void ne_someip_looper_runable_task_destory(ne_someip_looper_runnable_t* runable);
/** ne_someip_looper_post
 * @looper:
 *
 * 将一个ne_someip_looper_runnable_t提交给ne_someip_looper
 * 将在ne_someip_looper的主循环函数中执行这个ne_someip_looper_runnable_t对象
 * 返回值仅代表“提交”结果，不表示ne_someip_looper_runnable_t对象的执行结果
 *
 * Returns: 0:成功, -1: runable list加入失败, -2:唤醒失败
 */
int32_t ne_someip_looper_post(ne_someip_looper_t* looper, ne_someip_looper_runnable_t *runnable);
/** ne_someip_looper_self
 *
 * 取回当前线程的ne_someip_looper对象指针
 * 若当前线程没有存储ne_someip_looper对象，返回NULL
 *
 * Returns: 成功返回ne_someip_looper对象指针，失败返回NULL
 */
ne_someip_looper_t* ne_someip_looper_self();

/* ne_someip_looper_timer_t */
/** ne_someip_looper_timer_quick_create
 * @run_func: 运行函数
 * @free_func: 释放函数
 * @user_data:
 *
 * 创建ne_someip_looper_timer_t对象，ne_someip_looper_timer_t对象初始引用计数为1
 * runnable对象生命期同ne_someip_looper_timer_t对象，在ne_someip_looper_timer_t对象销毁时一并销毁。
 *
 * Returns: 返回ne_someip_looper_timer_t对象指针
 */
ne_someip_looper_timer_t* ne_someip_looper_timer_quick_create(void (*run_func) (void *user_data), void (*free_func) (void *user_data), void *user_data);
/** ne_someip_looper_timer_new
 * @runnable: 通过调用runnable对象的run函数，通知timeout到达事件
 *
 * 创建ne_someip_looper_timer_t对象，ne_someip_looper_timer_t对象初始引用计数为1
 * runnable对象生命期同ne_someip_looper_timer_t对象，在ne_someip_looper_timer_t对象销毁时一并销毁。
 *
 * Returns: 返回ne_someip_looper_timer_t对象指针
 */
ne_someip_looper_timer_t* ne_someip_looper_timer_new(ne_someip_looper_timer_runnable_t *runnable);
/** ne_someip_looper_timer_ref
 * @timer:
 *
 * 增加ne_someip_looper_timer_t对象的引用计数
 *
 * Returns: 返回ne_someip_looper_timer_t对象指针
 */
ne_someip_looper_timer_t* ne_someip_looper_timer_ref(ne_someip_looper_timer_t *timer);
/** ne_someip_looper_timer_unref
 * @timer:
 *
 * 减少ne_someip_looper_timer_t对象的引用计数
 *
 * Returns: 返回ne_someip_looper_timer_t对象指针
 */
void ne_someip_looper_timer_unref(ne_someip_looper_timer_t *timer);
/** ne_someip_looper_timer_refresh
 * @timer:
 * @type: timer类型，参考ne_someip_looper_timer_tType定义
 * @time_msec: timer的timeout时间，具体含义依据type有不同:
 *		type为NE_LOOPER_TIMER_TYPE_ABSOLUTE时，将在time_msec这个时刻，触发timeout事件
 *		type为NE_LOOPER_TIMER_TYPE_INTERVAL_ONE_SHOT时，从当前时刻，经过time_msec时间之后，触发一个timeout事件
 *		type为NE_LOOPER_TIMER_TYPE_INTERVAL_CYCLE时，从当前时刻开始，每time_msec时间为一个周期，触发一个timeout事件
 *
 * 刷新计时器，如果计时器还未启动，则启动计时器。如果已经启动，则更新timeout时间
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_looper_timer_refresh(ne_someip_looper_timer_t * timer, int32_t type, int64_t time_msec);
/** ne_someip_looper_timer_start
 * @timer:
 * @type: timer类型，参考ne_someip_looper_timer_tType定义
 * @time_msec: timer的timeout时间，具体含义依据type有不同:
 *		type为NE_LOOPER_TIMER_TYPE_ABSOLUTE时，将在time_msec这个时刻，触发timeout事件
 *		type为NE_LOOPER_TIMER_TYPE_INTERVAL_ONE_SHOT时，从当前时刻，经过time_msec时间之后，触发一个timeout事件
 *		type为NE_LOOPER_TIMER_TYPE_INTERVAL_CYCLE时，从当前时刻开始，每time_msec时间为一个周期，触发一个timeout事件
 *
 * 启动计时器
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_looper_timer_start(ne_someip_looper_timer_t *timer, int32_t type, int64_t time_msec);
/** ne_someip_looper_timer_stop
 * @timer:
 *
 * 停止计时器
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns:
 */
void ne_someip_looper_timer_stop(ne_someip_looper_timer_t *timer);
/** ne_someip_looper_timer_state
 * @timer:
 *
 * 获取定时器状态
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 定时器状态，参考ne_someip_looper_timer_state定义
 */
int32_t ne_someip_looper_timer_state(ne_someip_looper_timer_t *timer);

/* ne_someip_looper_io_source_t */
/** ne_someip_looper_io_source_quick_create
 * @fd: source将要监视的fd
 * @run_func: 运行函数
 * @free_func: 释放函数
 * @user_data:
 *
 * 创建ne_someip_looper_io_source_t对象，ne_someip_looper_io_source_t对象初始引用计数为1
 * runnable对象生命期同ne_someip_looper_io_source_t对象，在ne_someip_looper_io_source_t对象销毁时一并销毁。
 *
 * Returns: 返回ne_someip_looper_io_source_t对象指针
 */
ne_someip_looper_io_source_t* ne_someip_looper_io_source_quick_create(int32_t fd, void (*run_func) (ne_someip_looper_io_source_t* self, void *user_data, int32_t condition), void (*free_func) (void *user_data), void *user_data);
/** ne_someip_looper_io_source_new
 * @fd: source将要监视的fd
 * @runnable: 通过调用runnable对象的run函数，通知fd有新的事件
 *
 * 创建ne_someip_looper_io_source_t对象，ne_someip_looper_io_source_t对象初始引用计数为1
 * runnable对象生命期同ne_someip_looper_io_source_t对象，在ne_someip_looper_io_source_t对象销毁时一并销毁。
 *
 * Returns: 返回ne_someip_looper_io_source_t对象指针
 */
ne_someip_looper_io_source_t* ne_someip_looper_io_source_new(int32_t fd, ne_someip_looper_io_source_runnable_t *runnable);
/** ne_someip_looper_io_source_t
 * @source:
 *
 * 增加ne_someip_looper_io_source_tRef对象的引用计数
 *
 * Returns: 返回ne_someip_looper_io_source_t对象指针
 */
ne_someip_looper_io_source_t* ne_someip_looper_io_source_ref(ne_someip_looper_io_source_t* source);
/** ne_someip_looper_io_source_unref
 * @source:
 *
 * 减少ne_someip_looper_io_source_unref对象的引用计数
 *
 * Returns:
 */
void ne_someip_looper_io_source_unref(ne_someip_looper_io_source_t *soruce);
/** ne_someip_looper_io_source_start_by_runable
 * @soruce:
 * @condition: 监视的事件类型，参考ne_someip_looper_io_source_condition_t定义
 *
 * 在当前looper中创建runable，用来启动source监视
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_looper_io_source_start_by_runable(ne_someip_looper_io_source_t *soruce, int32_t condition);
/** ne_someip_looper_io_source_start_by_runable
 * @soruce:
 * @condition: 监视的事件类型，参考ne_someip_looper_io_source_condition_t定义
 *
 * 在当前looper中创建runable，用来停止source监视
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_looper_io_source_stop_by_runable(ne_someip_looper_io_source_t *soruce);
/** ne_someip_looper_io_source_add_condition
 * @soruce:
 * @condition: 监视的事件类型，参考ne_someip_looper_io_source_condition_t定义
 *
 * 追加监听类型
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_looper_io_source_add_condition(ne_someip_looper_io_source_t *source, int32_t add_condition);

/** ne_someip_looper_io_source_remove_condition
 * @soruce:
 * @condition: 监视的事件类型，参考ne_someip_looper_io_source_condition_t定义
 *
 * 移除监听类型
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_looper_io_source_remove_condition(ne_someip_looper_io_source_t *source, int32_t remove_condition);
/** ne_someip_looper_io_source_start
 * @soruce:
 * @condition: 监视的事件类型，参考ne_someip_looper_io_source_condition_t定义
 *
 * 启动source监视
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: 0:成功, -1: 失败
 */
int32_t ne_someip_looper_io_source_start(ne_someip_looper_io_source_t *soruce, int32_t condition);
/** ne_someip_looper_io_source_stop
 * @soruce:
 *
 * 停止source监视
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns:
 */
void ne_someip_looper_io_source_stop(ne_someip_looper_io_source_t *soruce);
/** ne_someip_looper_io_source_stateget
 * @source:
 *
 * 获取source对象状态
 * 本函数只能在ne_someip_looper线程中调用，否则返回失败
 *
 * Returns: source对象状态，参考ne_someip_looper_io_source_state_t定义
 */
int32_t ne_someip_looper_io_source_state(ne_someip_looper_io_source_t *soruce);

/** ne_someip_looper_time_get_timespec
 * 根据当前时间取得time
 */
void ne_someip_looper_time_get_timespec(struct timespec* time,  uint32_t msec);
/** ne_someip_looper_time_set_timespec
 * 将msec时间转为time时间
 */
void ne_someip_looper_time_set_timespec(struct timespec* time,  uint32_t msec);
/** ne_someip_looper_time_sub_timespec
 * 计算两个time之间的时间差
 */
long ne_someip_looper_time_sub_timespec(struct timespec* time1, struct timespec* time2);


#ifdef  __cplusplus
}
#endif
#endif // NE_SOMEIP_LOOPER_H
