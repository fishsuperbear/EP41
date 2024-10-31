#ifndef HW_HAL_IMPL_LOG_H
#define HW_HAL_IMPL_LOG_H

#include "hw_hal_api.h"
#include "hw_hal_impl_baseinc.h"

/*
* The implement inside will check init the hal impl log.
*/
struct hw_plat_logcontext_t* internal_get_plogcontext_hw_hal();

#define HW_HAL_IMPL_LOGRINGBUFFER_BYTECOUNT				0x2000000

struct hw_impl_hallogenv
{
	/*
	* 0 means has not init.
	* 1 means has init.
	*/
	u32								binit;
	struct hw_plat_logcontext_t		logcontext;
	/*
	* The offset of logringbuffer.
	*/
	struct hw_atomic_u32_t			atomic_offset;
	char							logringbuffer[HW_HAL_IMPL_LOGRINGBUFFER_BYTECOUNT];
};

#define HW_HAL_LOG(level,...)		do \
	{	\
		struct hw_plat_loghead_t _____internal_head;	\
		_____internal_head.func = __FUNCTION__;	\
		_____internal_head.file = __FILE__;	\
		_____internal_head.line = __LINE__;	\
		hw_plat_logoutput(internal_get_plogcontext_hw_hal(), level, &_____internal_head, NULL, __VA_ARGS__);	\
	} while(0)
#define HW_HAL_LOG_DEBUG(...)		HW_HAL_LOG(HW_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define HW_HAL_LOG_TRACE(...)		HW_HAL_LOG(HW_LOG_LEVEL_TRACE, __VA_ARGS__)
#define HW_HAL_LOG_INFO(...)		HW_HAL_LOG(HW_LOG_LEVEL_INFO, __VA_ARGS__)
#define HW_HAL_LOG_WARN(...)		HW_HAL_LOG(HW_LOG_LEVEL_WARN, __VA_ARGS__)
#define HW_HAL_LOG_ERR(...)			HW_HAL_LOG(HW_LOG_LEVEL_ERR, __VA_ARGS__)
#define HW_HAL_LOG_UNMASK(...)		HW_HAL_LOG(HW_LOG_LEVEL_UNMASK, __VA_ARGS__)
#define HW_HAL_LOG_FATAL(...)		HW_HAL_LOG(HW_LOG_LEVEL_FATAL, __VA_ARGS__)

#endif
