#ifndef HW_NVMEDIA_LOG_IMPL_H
#define HW_NVMEDIA_LOG_IMPL_H

#include "hw_nvmedia_def_impl.h"

/*
* The implement inside will check init the nvmedia impl log.
*/
struct hw_plat_logcontext_t* internal_get_plogcontext_nvmedia();

#define HW_NVMEDIA_LOG(level,...)	\
	do {	\
		struct hw_plat_loghead_t _____internal_head;	\
		_____internal_head.func = __FUNCTION__;	\
		_____internal_head.file = __FILE__;	\
		_____internal_head.line = __LINE__;	\
		hw_plat_logoutput(internal_get_plogcontext_nvmedia(), level, &_____internal_head, NULL, __VA_ARGS__);	\
	} while(0)
#define HW_NVMEDIA_LOG_DEBUG(...)		HW_NVMEDIA_LOG(HW_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define HW_NVMEDIA_LOG_TRACE(...)		HW_NVMEDIA_LOG(HW_LOG_LEVEL_TRACE, __VA_ARGS__)
#define HW_NVMEDIA_LOG_INFO(...)		HW_NVMEDIA_LOG(HW_LOG_LEVEL_INFO, __VA_ARGS__)
#define HW_NVMEDIA_LOG_WARN(...)		HW_NVMEDIA_LOG(HW_LOG_LEVEL_WARN, __VA_ARGS__)
#define HW_NVMEDIA_LOG_ERR(...)			HW_NVMEDIA_LOG(HW_LOG_LEVEL_ERR, __VA_ARGS__)
#define HW_NVMEDIA_LOG_UNMASK(...)		HW_NVMEDIA_LOG(HW_LOG_LEVEL_UNMASK, __VA_ARGS__)
#define HW_NVMEDIA_LOG_FATAL(...)		HW_NVMEDIA_LOG(HW_LOG_LEVEL_FATAL, __VA_ARGS__)

#define RET_HW_RET_S32_COMMON_UNEXPECTED	\
	do {	\
		HW_NVMEDIA_LOG_UNMASK("COMMON UNEXPECTED!\r\n");	\
		return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_COMMON_UNEXPECTED);	\
	} while (0)

#define CHK_LOG_SENTENCE_HW_RET_S32(retsentence)	\
	do {	\
		hw_ret_s32 __inner_ret__ = (retsentence);	\
		if (__inner_ret__ != 0) {	\
			HW_NVMEDIA_LOG_UNMASK("CHECK_LOG_HW_RET_S32 ret=0x%x, desc[%s]\r\n", __inner_ret__, hw_ret_s32_getdesc(__inner_ret__));	\
			return __inner_ret__;	\
		}	\
	} while (0)

/*
* Nvmedia log hw_s32 macros.
*/
#define CHK_PTR_AND_RET_S32(ptr, api) \
	do {	\
		if ((ptr) == NULL) { \
			HW_NVMEDIA_LOG_UNMASK("%s failed\r\n", (api)); \
			return HW_RET_S32_NVMEDIA_SIPLSTATUS(NVSIPL_STATUS_OUT_OF_MEMORY); \
		}	\
	} while(0)
#define CHK_PTR_AND_RET_S32_BADARG(ptr, name) \
	do {	\
		if ((ptr) == NULL) { \
			HW_NVMEDIA_LOG_UNMASK("%s is null\n", (name)); \
			return HW_RET_S32_NVMEDIA_SIPLSTATUS(NVSIPL_STATUS_BAD_ARGUMENT); \
		}	\
	} while(0)
#define CHK_SIPLSTATUS_SENTENCE_AND_RET_S32(statussentence, api) \
	do {	\
		SIPLStatus __inner_status__ = (statussentence);	\
		if (__inner_status__ != NVSIPL_STATUS_OK) { \
			HW_NVMEDIA_LOG_UNMASK("%s failed, siplstatus: %u\r\n", (api), __inner_status__); \
			return HW_RET_S32_NVMEDIA_SIPLSTATUS(__inner_status__); \
		}	\
	} while(0)
#define CHK_NVSCISTATUS_SENTENCE_AND_RET_S32(nvscistatussentence, api) \
	do {	\
		NvSciError __inner_status__ = (nvscistatussentence);	\
		if (__inner_status__ != NvSciError_Success) { \
			HW_NVMEDIA_LOG_UNMASK("%s failed, nvscistatus: %u\r\n", (api), __inner_status__); \
			return HW_RET_S32_NVMEDIA_NVSCISTATUS(__inner_status__); \
		}	\
	} while(0)
#define CHK_NVSCICONNECT_SENTENCE_AND_RET_S32(nvscistatussentence, eventtype, api)	\
	do {	\
		NvSciError __inner_status__ = (nvscistatussentence);	\
		if (__inner_status__ != NvSciError_Success) { \
			HW_NVMEDIA_LOG_UNMASK("%s failed, connect nvscistatus: %u\r\n", (api), __inner_status__); \
			return HW_RET_S32_NVMEDIA_NVSCISTATUS(__inner_status__); \
		}	\
		if (eventtype != NvSciStreamEventType_Connected) {	\
			HW_NVMEDIA_LOG_UNMASK("%s connected but didn't receive connected event.\r\n", (api));	\
			return HW_RET_S32_HW_HAL_NVMEDIA(HW_RET_S32_CODE_HW_HAL_NVMEDIA_CONNECT_DID_NOT_RECEIVE_CONNECTED_EVENT);	\
		}	\
	} while(0)

/*
* You may change the value.
*/
#define HW_NVMEDIA_IMPL_LOGRINGBUFFER_BYTECOUNT				0x2000000

struct hw_impl_nvmedialogenv
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
	char							logringbuffer[HW_NVMEDIA_IMPL_LOGRINGBUFFER_BYTECOUNT];
};

#endif
