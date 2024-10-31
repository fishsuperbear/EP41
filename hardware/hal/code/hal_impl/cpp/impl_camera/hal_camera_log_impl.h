#ifndef HAL_CAMERA_LOG_IMPL_H
#define HAL_CAMERA_LOG_IMPL_H

#include "hal_camera_baseinc_impl.h"
#include "hal_camera_def_impl.h"

/*
* The implement inside will check init the nvmedia impl log.
*/
struct hw_plat_logcontext_t* internal_get_plogcontext_halcamera();

#define HAL_CAMERA_LOG(level,...)   \
    do {    \
        struct hw_plat_loghead_t _____internal_head;    \
        _____internal_head.func = __FUNCTION__; \
        _____internal_head.file = __FILE__; \
        _____internal_head.line = __LINE__; \
        hw_plat_logoutput(internal_get_plogcontext_halcamera(), level, &_____internal_head, NULL, __VA_ARGS__); \
    } while(0)
#define HAL_CAMERA_LOG_DEBUG(...)       HAL_CAMERA_LOG(HW_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define HAL_CAMERA_LOG_TRACE(...)       HAL_CAMERA_LOG(HW_LOG_LEVEL_TRACE, __VA_ARGS__)
#define HAL_CAMERA_LOG_INFO(...)        HAL_CAMERA_LOG(HW_LOG_LEVEL_INFO, __VA_ARGS__)
#define HAL_CAMERA_LOG_WARN(...)        HAL_CAMERA_LOG(HW_LOG_LEVEL_WARN, __VA_ARGS__)
#define HAL_CAMERA_LOG_ERR(...)         HAL_CAMERA_LOG(HW_LOG_LEVEL_ERR, __VA_ARGS__)
#define HAL_CAMERA_LOG_UNMASK(...)      HAL_CAMERA_LOG(HW_LOG_LEVEL_UNMASK, __VA_ARGS__)
#define HAL_CAMERA_LOG_FATAL(...)       HAL_CAMERA_LOG(HW_LOG_LEVEL_FATAL, __VA_ARGS__)

/*
* You may change the value.
*/
#define HAL_CAMERA_IMPL_LOGRINGBUFFER_BYTECOUNT             0x2000000
/*
* We at most can register the specific number data callback type.
* The count is properly bigger than the supported datacb type.
* Currently, one type we can only register one data callback.
*/
#define CAMERA_DEVICE_DATACB_TYPE_MAXCOUNT                  16

struct hw_impl_halcameralogenv
{
    /*
    * 0 means has not init.
    * 1 means has init.
    */
    u32                             binit;
    struct hw_plat_logcontext_t     logcontext;
    /*
    * The offset of logringbuffer.
    */
    struct hw_atomic_u32_t          atomic_offset;
    char                            logringbuffer[HAL_CAMERA_IMPL_LOGRINGBUFFER_BYTECOUNT];
};

#endif
