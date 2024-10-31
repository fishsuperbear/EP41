#ifndef NETA_KLOG_H
#define NETA_KLOG_H

#include "neta_kerr.h"

#ifndef NETA_BUDDY_MAXSIZE
// 4MB equal to the default buddy system max continuous size
#define NETA_BUDDY_MAXSIZE                      0x400000
#endif

/*
* Get bits h:low of u64 u.
*/
#define U64BITH2L(u, h, low)		((((u64)(u))>>(low))&(((h)-(low)+1==64)?(~0):(~((u64)-1<<((h)-(low)+1)))))
#define U32BITH2L(u, h, low)		((((u32)(u))>>(low))&(((h)-(low)+1==32)?(~0):(~((u32)-1<<((h)-(low)+1)))))
#define U16BITH2L(u, h, low)		((((u16)(u))>>(low))&(((h)-(low)+1==16)?(~0):(~((u16)-1<<((h)-(low)+1)))))
#define U8BITH2L(u, h, low)			((((u8)(u))>>(low))&(((h)-(low)+1==8)?(~0):(~((u8)-1<<((h)-(low)+1)))))

#define NETA_DMB    \
    do {    \
        __asm__ __volatile__("dmb st" ::: "memory");    \
    } while (0)

#define NETA_KLOG_HEAD_TASKSTRUCT_FORMAT       "tgid:%d-pid:%d-cpu:%d"
#define NETA_KLOG_HEAD_TASKSTRUCT_PARA         current->tgid, current->pid, current->cpu

#define NETA_KLOG_HEAD_FORMAT                  " ("NETA_KLOG_HEAD_TASKSTRUCT_FORMAT" %s #%d) "
#define NETA_KLOG_HEAD_PARA                    NETA_KLOG_HEAD_TASKSTRUCT_PARA, __FUNCTION__, __LINE__

#define NETA_KLOG_INFO( fmt, args... )  \
    do {    \
        pr_info( "info" NETA_KLOG_HEAD_FORMAT fmt, NETA_KLOG_HEAD_PARA, ##args ); \
    } while (0)

#define NETA_KLOG_ERR( fmt, args... )   \
    do {    \
        pr_err( "error" NETA_KLOG_HEAD_FORMAT fmt, NETA_KLOG_HEAD_PARA, ##args ); \
    } while (0)

#define NETA_KLOG_INFO_RATELIMITED( fmt, args... )  \
    do {    \
        pr_info_ratelimited( "info" NETA_KLOG_HEAD_FORMAT fmt, NETA_KLOG_HEAD_PARA, ##args ); \
    } while (0)

#define NETA_KLOG_ERR_RATELIMITED( fmt, args... )   \
    do {    \
        pr_err_ratelimited( "error" NETA_KLOG_HEAD_FORMAT fmt, NETA_KLOG_HEAD_PARA, ##args ); \
    } while (0)

#define NETA_KLOG_INFO_HEXDUMP( addr, size) \
    do {    \
        print_hex_dump(KERN_INFO, "hex_dump:", DUMP_PREFIX_NONE, 32, 4, addr, size, true);  \
    } while (0)

#define NETA_KLOG_INFO_TRACE    \
    do {    \
        NETA_KLOG_INFO("run here!\n");\
    } while (0)

#define NETA_KLOG_UNEXPECTED_RUN_HERE   \
    do {    \
        NETA_KLOG_ERR("Unexpected run here!\n"); \
    } while (0)

#define RET_UNEXPECTED_RUN_HERE   \
    do {    \
        NETA_KLOG_ERR("Unexpected run here!\n"); \
        return NETA_KERR_COMMON_UNEXPECTED;    \
    } while (0)

#define CHK_PTR_AND_RET_S32(ptr, api)   \
    do {    \
        if (unlikely(ptr == NULL)){   \
            NETA_KLOG_ERR("run %s return NULL!\n", (api)); \
            return NETA_KERR_COMMON_RETURN_NULL_PTR;    \
        }   \
    } while (0)

#define CHK_SENTENCE_RET(retsentence, api)  \
    do {	\
		s32 __inner_ret__ = (retsentence);	\
		if (unlikely(__inner_ret__ != 0)) {	\
			NETA_KLOG_ERR("CHK_SENTENCE_RET [%s] ret=%d[%s]\n", (api), __inner_ret__, neta_kerr_info(__inner_ret__));	\
			return __inner_ret__;	\
		}	\
	} while (0)

static inline u64 neta_gettscns(void)
{
    u64 tsc;
    __asm__ __volatile__ ("mrs %[tsc], cntvct_el0" : [tsc] "=r" (tsc));
    return (u64)(tsc * 32);
}

// set it 0 or 1, 1 means using preempt_disable when there's a lot of tag dirty data to read
#define NETA_ENABLE_HIGH_PERFORMANCE_MODE   1

#if (NETA_ENABLE_HIGH_PERFORMANCE_MODE == 1)
// set it 0 or 1
#define NETA_ENABLE_LOG_OF_HIGH_PERFORMANCE_MODE    0
#else
// do not change the define
#define NETA_ENABLE_LOG_OF_HIGH_PERFORMANCE_MODE    0
#endif

// set it 0 or 1, set it to 1 for debug only, for log only
#define NETA_KLOG_ENABLE_DEBUG_LOG      0

#if (NETA_KLOG_ENABLE_DEBUG_LOG == 1)
// set it 0 or 1
#define NETA_KLOG_ENABLE_EXTRA_DEBUG_LOG      0
// set it 0 or 1
#define NETA_KLOG_ENABLE_COMBINENS_DEBUG      1
#else
// do not change the define
#define NETA_KLOG_ENABLE_EXTRA_DEBUG_LOG      0
// do not change the define
#define NETA_KLOG_ENABLE_COMBINENS_DEBUG      0
#endif

// set it 0 or 1, set it to 1 for debug only, for logic
#define NETA_ENABLE_DEBUG_LOGIC         0

#endif
