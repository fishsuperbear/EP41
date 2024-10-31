#ifndef NETA_KERR_H
#define NETA_KERR_H

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/printk.h>
#include <linux/dma-buf.h>
#include <linux/pagemap.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/version.h>
#include <linux/errno.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <uapi/linux/sched/types.h>
#include <linux/sched/types.h>
#include <linux/sched.h>

#ifndef s8
typedef signed char         s8;
#endif
#ifndef s16
typedef short               s16;
#endif
#ifndef s32
typedef int                 s32;
#endif
#ifndef s64
typedef long long           s64;
#endif
#ifndef u8
typedef unsigned char       u8;
#endif
#ifndef u16
typedef unsigned short      u16;
#endif
#ifndef u32
typedef unsigned int        u32;
#endif
#ifndef u64
typedef unsigned long long  u64;
#endif

#ifndef STATIC_ASSERT
#define STATIC_ASSERT(truecond)		static_assert (truecond, "error")
#endif

STATIC_ASSERT(sizeof(u8) == 1);
STATIC_ASSERT(sizeof(u16) == 2);
STATIC_ASSERT(sizeof(u32) == 4);
STATIC_ASSERT(sizeof(u64) == 8);

enum NETA_KERR
{
    NETA_KERR_BEGIN = -1024,

    NETA_KERR_COMMON_UNEXPECTED,
    NETA_KERR_COMMON_RETURN_NULL_PTR,

    // log block part
    NETA_KERR_LOGBLOCKMAXCOUNT_PERBUDDY_POWER_CHECK,
    NETA_KERR_LOGBLOCKBUDDYMAXCOUNT_PERGROUP_POWER_CHECK,
    NETA_KERR_LOGBLOCKGROUP_NOTINIT,
    NETA_KERR_LOGBLOCKGROUP_BQUIT_IS_1,
    NETA_KERR_LOGBLOCKGROUP_NEEDNEW_INDEX_WRONG,
    NETA_KERR_LOGBLOCKGROUP_EXCEEDMAXCOUNT,
    NETA_KERR_LOGBLOCKBUDDY_BRESERVED_1_NOT_SUPPORT,
    NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLPRODUCERFUNC_WHENCONSUMER,
    NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLCONSUMERFUNC_WHENPRODUCER,
    NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLFINISHLOGBLOCK,
    NETA_KERR_LOGBLOCKUSER_THREADCOOKIE_ALREADY_NOT_0,
    NETA_KERR_LOGBLOCKUSER_THREADCOOKIE_IS_0,
    NETA_KERR_LOGBLOCKUSER_LOGBLOCK_DMABUFEXPORT_FAIL,
    NETA_KERR_LOGBLOCKUSER_LOGBLOCKGROUPSHARE_DMABUFEXPORT_FAIL,
    NETA_KERR_LOGBLOCKUSER_LOGBLOCKGROUP_DMABUFEXPORT_FAIL,
    NETA_KERR_LOGBLOCKUSER_TRYPUT_CHECKREFCOUNT_NOT_2,
    NETA_KERR_LOGBLOCKUSER_CONSUMERCOUNT_CHECK_FAIL,
    NETA_KERR_LOGBLOCKUSER_LOGBLOCKCOOKIE_IS_0,
};

const char* neta_kerr_info(s32 i_s32);

#endif
