#ifndef INCLUDE_OSAL_PERFTOOLKIT_PT_TRACE_API_H_
#define INCLUDE_OSAL_PERFTOOLKIT_PT_TRACE_API_H_

#include "pt_trace_core.h"

#ifndef PT_TRACE_THIS_TAG
#error ERROR: PT_TRACE_THIS_TAG must be defined before include "pt_trace_api.h"
#endif

#define PT_TRACE_IS_ENABLED() \
(PT_TRACE_UNLIKELY(g_pt_trace_enabled_flag) \
&& PT_TRACE_UNLIKELY(((PT_TRACE_THIS_TAG|PT_TRACE_FLAG_START) & g_pt_trace_enabled_tags) \
== (PT_TRACE_THIS_TAG|PT_TRACE_FLAG_START)))

/* 使用者需要在自己的Makefile或者.c/.cc/.cpp文件中定义PT_TRACE_IS_ENABLED这个宏，才能使能Trace功能 */
// #ifdef PT_TRACE_IS_ENABLED
// /* PT_TRACE_FILTER=name1:filter,name2:filter */
// /* 定义模块名，可通过环境变量设置过滤规则，每个模块至多定义一个，推荐一个so定义一次*/
// #define PT_TRACE_API_DEFINE_MODULE(name) PT_TRACE_CORE_DEFINE_MODULE(name)
// /* 定义模块过滤器，每个.c/.cc.cpp至多声明一个，id取值范围为0~63 */
// #define PT_TRACE_API_DECAL_FILTER(name, id) PT_TRACE_CORE_DECAL_FILTER(name, id)
/* 在线程Track上记录函数入口，普通字符串格式 */
#define PT_TRACE_BEGIN(str) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_BEGIN(str)
/* 在线程Track上记录函数入口，格式化字符串格式 */
#define PT_TRACE_BEGINF(format, ...) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_BEGINF(format, ##__VA_ARGS__)
/* 在线程Track上记录函数入口，格式化字符串格式 */
#define PT_TRACE_BEGINVF(format, arglist) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_BEGINVF(format, arglist)
/* 在线程Track上记录函数出口 */
#define PT_TRACE_END() if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_END()
/* 在线程Track上记录即时信息，普通字符串格式 */
#define PT_TRACE_INSTANT(str) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_INSTANT(str)
/* 在线程Track上记录即时信息，格式化字符串格式 */
#define PT_TRACE_INSTANTF(format, ...) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_INSTANTF(format, ##__VA_ARGS__)
/* 在线程Track上记录即时信息，格式化字符串格式 */
#define PT_TRACE_INSTANTVF(format, arglist) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_INSTANTVF(format, arglist)
/* 在name所在Track上记录异步动作开始，通过64bit的cookie和final关联 */
#define PT_TRACE_ASYNC_START(name, cookie) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_ASYNC_START(name, cookie)
/* 在name所在Track上记录异步动作结束，通过64bit的cookie和start关联 */
#define PT_TRACE_ASYNC_FINAL(name, cookie) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_ASYNC_FINAL(name, cookie)
/* 在name所在Track上记录64bit的整形值 */
#define PT_TRACE_COUNTER(name, value) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_COUNTER(name, value)
/* 在name所在Track上记录即时信息，普通字符串格式 */
#define PT_TRACE_INFO(name, str) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_INFO(name, str)
/* 在name所在Track上记录即时信息，格式化字符串格式 */
#define PT_TRACE_INFOF(name, format, ...) if (PT_TRACE_IS_ENABLED()) PT_TRACE_CORE_INFOF(name, format, ##__VA_ARGS__)
/* 在name所在Track上记录即时信息，格式化字符串格式 */
#define PT_TRACE_INFOVF(name, format, arglist) if (PT_TRACE_IS_ENABLED()) PT_TRACE_INFOVF(name, format, arglist)
#ifdef __cplusplus
// C++专用，利用RAII机制，自动记录PT_TRACE_BEGINVF和PT_TRACE_END
#define PT_TRACE_FUNCTION_AUTO(...) PtTraceCoreAutoScope PT_TRACE_CORE_UNIQUE_NAME(__VA_ARGS__)
struct PtTraceCoreAutoScope {
    PtTraceCoreAutoScope(const char* format, ...) {
        va_list arglist;
        va_start(arglist, format);
        PT_TRACE_BEGINVF(format, arglist);
        va_end(arglist);
    }
    ~PtTraceCoreAutoScope() {PT_TRACE_END();}
};
#endif
// #else
// #define PT_TRACE_BEGIN(str)
// #define PT_TRACE_BEGINF(format, ...)
// #define PT_TRACE_BEGINVF(format, arglist)
// #define PT_TRACE_END()
// #define PT_TRACE_INSTANT(str)
// #define PT_TRACE_INSTANTF(format, ...)
// #define PT_TRACE_INSTANTVF(format, arglist)
// #define PT_TRACE_ASYNC_START(name, cookie)
// #define PT_TRACE_ASYNC_FINAL(name, cookie)
// #define PT_TRACE_COUNTER(name, value)
// #define PT_TRACE_INFO(name, str)
// #define PT_TRACE_INFOF(name, format, ...)
// #define PT_TRACE_INFOVF(name, format, arglist)
// #define PT_TRACE_FUNCTION_AUTO(...)
// #endif

#endif  // INCLUDE_OSAL_PERFTOOLKIT_PT_TRACE_API_H_
