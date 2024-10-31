#ifndef HW_LIDAR_LOG_IMPL_H
#define HW_LIDAR_LOG_IMPL_H

#include "lidar/modules/common/hw_lidar_modules_common.h"

/*
 * The implement inside will check init the nvmedia impl log.
 */
struct hw_plat_logcontext_t *internal_get_plogcontext_nvmedia();

#define HW_NVMEDIA_LOG(level, ...)                                                                            \
	do                                                                                                        \
	{                                                                                                         \
		struct hw_plat_loghead_t _____internal_head;                                                          \
		_____internal_head.func = __FUNCTION__;                                                               \
		_____internal_head.file = __FILE__;                                                                   \
		_____internal_head.line = __LINE__;                                                                   \
		hw_plat_logoutput(internal_get_plogcontext_nvmedia(), level, &_____internal_head, NULL, __VA_ARGS__); \
	} while (0)

#define HW_LIDAR_LOG(level, format, ...)            \
	do                                              \
	{                                               \
		printf("%s(%d): ", __FUNCTION__, __LINE__); \
		printf(format, ##__VA_ARGS__);              \
	} while (0)

#define HW_LIDAR_LOG_DEBUG(...) HW_LIDAR_LOG(HW_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define HW_LIDAR_LOG_TRACE(...) HW_LIDAR_LOG(HW_LOG_LEVEL_TRACE, __VA_ARGS__)
#define HW_LIDAR_LOG_INFO(...) HW_LIDAR_LOG(HW_LOG_LEVEL_INFO, __VA_ARGS__)
#define HW_LIDAR_LOG_WARN(...) HW_LIDAR_LOG(HW_LOG_LEVEL_WARN, __VA_ARGS__)
#define HW_LIDAR_LOG_ERR(...) HW_LIDAR_LOG(HW_LOG_LEVEL_ERR, __VA_ARGS__)
#define HW_LIDAR_LOG_UNMASK(...) HW_LIDAR_LOG(HW_LOG_LEVEL_UNMASK, __VA_ARGS__)
#define HW_LIDAR_LOG_FATAL(...) HW_LIDAR_LOG(HW_LOG_LEVEL_FATAL, __VA_ARGS__)

#endif // HW_LIDAR_LOG_IMPL_H
