#ifndef HW_NVMEDIA_LOG_DEVICEBLOCK_IMPL_H
#define HW_NVMEDIA_LOG_DEVICEBLOCK_IMPL_H

#include "hw_nvmedia_log_impl.h"

/*
* The implement inside will check init the nvmedia impl log.
*/
struct hw_plat_logcontext_t* internal_get_plogcontext_nvmedia_deviceblock();

#define HW_NVMEDIA_DEVICEBLOCK_LOG(level,...)	\
	do {	\
		struct hw_plat_loghead_t _____internal_head;	\
		_____internal_head.func = __FUNCTION__;	\
		_____internal_head.file = __FILE__;	\
		_____internal_head.line = __LINE__;	\
		hw_plat_logoutput(internal_get_plogcontext_nvmedia_deviceblock(), level, &_____internal_head, NULL, __VA_ARGS__);	\
	} while(0)
#define HW_NVMEDIA_DEVICEBLOCK_LOG_DEBUG(...)				HW_NVMEDIA_DEVICEBLOCK_LOG(HW_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_LOG_TRACE(...)				HW_NVMEDIA_DEVICEBLOCK_LOG(HW_LOG_LEVEL_TRACE, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_LOG_INFO(...)				HW_NVMEDIA_DEVICEBLOCK_LOG(HW_LOG_LEVEL_INFO, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_LOG_WARN(...)				HW_NVMEDIA_DEVICEBLOCK_LOG(HW_LOG_LEVEL_WARN, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_LOG_ERR(...)					HW_NVMEDIA_DEVICEBLOCK_LOG(HW_LOG_LEVEL_ERR, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_LOG_UNMASK(...)				HW_NVMEDIA_DEVICEBLOCK_LOG(HW_LOG_LEVEL_UNMASK, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_LOG_FATAL(...)				HW_NVMEDIA_DEVICEBLOCK_LOG(HW_LOG_LEVEL_FATAL, __VA_ARGS__)

#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(level,...)	\
	do {	\
		hw_plat_logoutput(internal_get_plogcontext_nvmedia_deviceblock(), level, NULL, NULL, __VA_ARGS__);	\
	} while(0)
#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_DEBUG(...)		HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(HW_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_TRACE(...)		HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(HW_LOG_LEVEL_TRACE, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_INFO(...)			HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(HW_LOG_LEVEL_INFO, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_WARN(...)			HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(HW_LOG_LEVEL_WARN, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_ERR(...)			HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(HW_LOG_LEVEL_ERR, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_UNMASK(...)		HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(HW_LOG_LEVEL_UNMASK, __VA_ARGS__)
#define HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG_FATAL(...)		HW_NVMEDIA_DEVICEBLOCK_NOHEAD_LOG(HW_LOG_LEVEL_FATAL, __VA_ARGS__)

/*
* You may change the value.
*/
#define HW_NVMEDIA_DEVICEBLOCK_IMPL_LOGRINGBUFFER_BYTECOUNT				0x2000000

struct hw_impl_nvmediadeviceblocklogenv
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
	char							logringbuffer[HW_NVMEDIA_DEVICEBLOCK_IMPL_LOGRINGBUFFER_BYTECOUNT];
};

#endif
