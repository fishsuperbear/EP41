#include "hal_camera_log_impl.h"
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <mutex>

#define HALLOG_CTRL_NDOE_POLLTIME     1000000000
#define HALLOG_CTRL_LEVEL_MIN         1
#define HALLOG_CTRL_LEVEL_MAX         8

static std::mutex hal_camera_mutex;

static struct hw_impl_halcameralogenv _halcameralogenv =
{
    .binit = 0,
};

static s32 hal_camera_get_nodelevel(u32 *level)
{
    char hallog_ctrl_buffer[64] = {0};

    int fd = open("/proc/netahal/hallog_ctrl/hal_camera", O_RDONLY);
    if (fd < 0) {
        close(fd);
        return -1;
    }
    ssize_t bytesRead = read(fd, hallog_ctrl_buffer, sizeof(hallog_ctrl_buffer) - 1);
    if (bytesRead = 1) {
        hallog_ctrl_buffer[bytesRead] = '\0';
        *level = atoi(hallog_ctrl_buffer);
    } else {
        close(fd);
        return -2;
    }

    close(fd);

    return 0;
}

static s32 hal_camera_impl_log_init()
{
    if (HW_LIKELY(_halcameralogenv.binit == 0))
    {
        std::lock_guard<std::mutex> lock(hal_camera_mutex);
        if (_halcameralogenv.binit == 1) {
            return -1;
        }
        s32 ret;
        ret = hw_plat_logcontext_fill_bydefault(&_halcameralogenv.logcontext);
        if (ret < 0) {
            return -1;
        }
        strcpy(_halcameralogenv.logcontext.logoper.innerimpl.logdir, "./hallog/camera");
        u32 initvalue = 0;
        _halcameralogenv.logcontext.level = HW_LOG_LEVEL_INFO;
        ret = hw_plat_logcontext_fill_bufmode_logbuf(&_halcameralogenv.logcontext,
            _halcameralogenv.logringbuffer, HAL_CAMERA_IMPL_LOGRINGBUFFER_BYTECOUNT, HW_PLAT_LOGCONTEXT_LOGBUFLEVEL_DEFAULT,
            &_halcameralogenv.atomic_offset, &initvalue);
        if (ret < 0) {
            return -1;
        }
        ret = hw_plat_logcontext_init(&_halcameralogenv.logcontext);
        if (ret < 0) {
            return -1;
        }
        _halcameralogenv.binit = 1;
        return 0;
    }
    return -1;
}

struct hw_plat_logcontext_t* internal_get_plogcontext_halcamera()
{
    u32 levelvalue = 0;
    u64 get_time = 0;
    static u64 lasttime = 0;
    u64 currenttime = 0, difftime = 0;

    hw_plat_get_tsc_ns(&get_time);
    if (HW_UNLIKELY(_halcameralogenv.binit == 0))
    {
        lasttime = get_time;
        hal_camera_impl_log_init();

        if (hal_camera_get_nodelevel(&levelvalue)) {
            return &_halcameralogenv.logcontext;
        }

        if ((levelvalue >= HALLOG_CTRL_LEVEL_MIN) && (levelvalue <= HALLOG_CTRL_LEVEL_MAX) && (levelvalue != _halcameralogenv.logcontext.level)) {
            if (hw_plat_loglevel_set(&_halcameralogenv.logcontext, levelvalue)) {
                return &_halcameralogenv.logcontext;
            }
        }
    } else {
        currenttime = get_time;
        difftime = currenttime - lasttime;
        if (difftime > HALLOG_CTRL_NDOE_POLLTIME) {
            if (hal_camera_get_nodelevel(&levelvalue)) {
                return &_halcameralogenv.logcontext;
            }

            if ((levelvalue >= HALLOG_CTRL_LEVEL_MIN) && (levelvalue <= HALLOG_CTRL_LEVEL_MAX) && (levelvalue != _halcameralogenv.logcontext.level)) {
                if (hw_plat_loglevel_set(&_halcameralogenv.logcontext, levelvalue)) {
                    return &_halcameralogenv.logcontext;
                }
            }
        }
        lasttime = currenttime;
    }

    return &_halcameralogenv.logcontext;
}
