#include "hw_nvmedia_log_impl.h"
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <mutex>

#define HALLOG_CTRL_NDOE_POLLTIME     1000000000
#define HALLOG_CTRL_LEVEL_MIN         1
#define HALLOG_CTRL_LEVEL_MAX         8

static std::mutex hw_nvmedia_mutex;

const char* _plogsubdesc = "default";

static struct hw_impl_nvmedialogenv _nvmedialogenv =
{
    .binit = 0,
};

static s32 hw_nvmedia_get_nodelevel(u32 *level)
{
    char hallog_ctrl_buffer[64] = {0};

    int fd = open("/proc/netahal/hallog_ctrl/hw_nvmedia", O_RDONLY);
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

static s32 hw_nvmedia_impl_log_init()
{
    if (HW_LIKELY(_nvmedialogenv.binit == 0))
    {
        std::lock_guard<std::mutex> lock(hw_nvmedia_mutex);
        if (_nvmedialogenv.binit == 1) {
            return -1;
        }
        s32 ret;
        ret = hw_plat_logcontext_fill_bydefault(&_nvmedialogenv.logcontext);
        if (ret < 0) {
            return -1;
        }
        strcpy(_nvmedialogenv.logcontext.logoper.innerimpl.logdir, "./hallog/nvmedia/multiipc_");
#if (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_MULTIIPC_PRODUCER)
        _plogsubdesc = const_cast<char*>("main");
#elif (HW_NVMEDIA_PROJ == HW_NVMEDIA_PROJ_MULTIIPC_CONSUMER_CUDA)
        _plogsubdesc = const_cast<char*>("cuda");
#endif
        strcat(_nvmedialogenv.logcontext.logoper.innerimpl.logdir, _plogsubdesc);
        u32 initvalue = 0;
        _nvmedialogenv.logcontext.level = HW_LOG_LEVEL_ERR;
        ret = hw_plat_logcontext_fill_bufmode_logbuf(&_nvmedialogenv.logcontext,
            _nvmedialogenv.logringbuffer, HW_NVMEDIA_IMPL_LOGRINGBUFFER_BYTECOUNT, HW_PLAT_LOGCONTEXT_LOGBUFLEVEL_DEFAULT,
            &_nvmedialogenv.atomic_offset, &initvalue);
        if (ret < 0) {
            return -1;
        }
        ret = hw_plat_logcontext_init(&_nvmedialogenv.logcontext);
        if (ret < 0) {
            return -1;
        }
        _nvmedialogenv.binit = 1;
        return 0;
    }
    return -1;
}

struct hw_plat_logcontext_t* internal_get_plogcontext_nvmedia()
{
    u32 levelvalue = 0;
    u64 get_time = 0;
    static u64 lasttime = 0;
    u64 currenttime = 0, difftime = 0;

    hw_plat_get_tsc_ns(&get_time);
    if (HW_UNLIKELY(_nvmedialogenv.binit == 0))
    {
        lasttime = get_time;
        hw_nvmedia_impl_log_init();

        if (hw_nvmedia_get_nodelevel(&levelvalue)) {
            return &_nvmedialogenv.logcontext;
        }

        if ((levelvalue >= HALLOG_CTRL_LEVEL_MIN) && (levelvalue <= HALLOG_CTRL_LEVEL_MAX) && (levelvalue != _nvmedialogenv.logcontext.level)) {
            if (hw_plat_loglevel_set(&_nvmedialogenv.logcontext, levelvalue)) {
                return &_nvmedialogenv.logcontext;
            }
        }
    } else {
        currenttime = get_time;
        difftime = currenttime - lasttime;
        if (difftime > HALLOG_CTRL_NDOE_POLLTIME) {
            if (hw_nvmedia_get_nodelevel(&levelvalue)) {
                return &_nvmedialogenv.logcontext;
            }

            if ((levelvalue >= HALLOG_CTRL_LEVEL_MIN) && (levelvalue <= HALLOG_CTRL_LEVEL_MAX) && (levelvalue != _nvmedialogenv.logcontext.level)) {
                if (hw_plat_loglevel_set(&_nvmedialogenv.logcontext, levelvalue)) {
                    return &_nvmedialogenv.logcontext;
                }
            }
        }
        lasttime = currenttime;
    }

    return &_nvmedialogenv.logcontext;
}
