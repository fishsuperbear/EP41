#ifndef NETA_LOG_H
#define NETA_LOG_H

#include "neta_node_common.h"
// should be exactly the same file as file release to upper level
#include "logblockwriter.h"
// should be exactly the same file as file release to upper level
#include "logblockreader.h"
// shuold be exactly the same file as file used by lognode user impl
#include "neta_lognode_ioctl.h"

#include "neta_lognode.h"
#include "neta_logblock.h"
#include "neta_lognode_ioctl_impl.h"
#include "neta_group_manager.h"
/*
* No neta_log_init function due to we init it in neta_log.c GNC struct init
* code style.
*/
typedef struct neta_log
{
    /*
    * Only increase, never minus. 
    * Value is 1 ~ NETA_LOGBLOCKGROUP_MAXCOUNT after sucessfully init.
    */
    u32                         groupcount;
    /*
    * Only use it when need to malloc new group.
    */
    struct mutex                mutexnewgroup;
    /*
    * Set when neta_logblockgroup_create.
    */
    neta_logblockgroup*         parraypgroup[NETA_LOGBLOCKGROUP_MAXCOUNT];
    /*
    * Set when neta_logblockgroup_create.
    * Every page* is of every neta_logblockgroup.
    * You should use free_pages to free the page* element.
    * The size of the page is certain sizeof(neta_logblockgroup).
    */
    struct page*                parrayppagegroup[NETA_LOGBLOCKGROUP_MAXCOUNT];
    /*
    * Set when neta_logblockgroup_create.
    */
    neta_logblockgroupshare*    parraypgroupshare[NETA_LOGBLOCKGROUP_MAXCOUNT];
    /*
    * Set when neta_logblockgroup_create.
    * Every page* is of every neta_logblockgroupshare.
    * You should use free_pages to free the page* element.
    * The size of the page is certain sizeof(neta_logblockgroupshare).
    */
    struct page*                parrayppagegroupshare[NETA_LOGBLOCKGROUP_MAXCOUNT];
    /*
    * 0 ~ groupcount-1
    * For neta_logblock_get function use.
    * Find vacant log block of the specific expectedgroupindex block group first.
    * When cannot find vacant log block of the expectedgroupindex, logic will check 
    * every log block group one by one until all of the current log group has been 
    * checked. And then if still cannot find vacant log block, logic will malloc a 
    * new log block group.
    */
    u32                         expectedgroupindex;
    /*
    * The writer process count. We do not count writer thread count because the writer
    * thread count is properly incorrect.
    */
    u32                         writerprocesscount;
    /*
    * The reader thread count.
    */
    u32                         readerthreadcount;
    /*
    * Begin from 0, it may exceed the readerthreadcount.
    * Every new reader thread create, it will plus one.
    */
    u32                         expectedreaderthreadid;
    /*
    * Currently, it will wake up all of the log block consumer thread.
    * It is triggered by logblock producer tagdirty ioctl.
    */
    wait_queue_head_t           tagdirtywait;
    /*
    * Default is 0, set to 1 when tag dirty ioctl, reset to 0 when one of the 
    * log block consumer thread has been wake up.
    */
    u32                         btagdirty;
    /*
    * 0 or 1, 1 means it is running rmmod operation.
    */
    u32                         bquit;
    /*
    * Last iterate time, iterate every log block wp and rp every 
    * INTERNAL_NETA_LOGREADERTHREAD_ITERATE_TIME_NS ns.
    */
    u64                         lastiteratetimens;
    /*
    * The notify 'already half' logblock pointer.
    */
    neta_logblock*              alreadyhalfplogblockarray[NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT];
    /*
    * Tag whether the alreadyhalfplogblockarray is valid or not.
    */
    u32                         breadyarray[NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT];
    /*
    * Write pointer.
    * Use atomic add to the value, you need to do & (NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT -1)
    * operation to get the exact array index of alreadyhalfplogblockarray.
    */
    u32                         wpalreadyhalf;
    /*
    * Read pointer.
    * Use atomic add to the value, you need to do & (NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT -1)
    * operation to get the exact array index of alreadyhalfplogblockarray.
    */
    u32                         rpalreadyhalf;
    /*
    * Already half counter. Do not use atomic operation to the variable due to performance.
    */
    u32                         alreadyhalfcounter;
    /*
    * 0 or 1, 1 means now is clearing group.
    */
    u32                         bclearinggroup;
    /*
    * The next group index to clear when no enough log block.
    */
    u32                         grouptoclearindex;
} neta_log;

extern neta_log         __neta_log;

/*
* return 0 means has init.
* return not 0 means has not init.
*/
static inline s32 __always_inline neta_logblock_checkhasinit(void)
{
    if (unlikely(__neta_log.groupcount == 0)) {
        return -1;
    }
    return 0;
}

/*
* It will check whether has init log block group has init.
* If it has not inited, it will return -1.
*/
#define NETA_LOGBLOCKGROUP_CHECKHASINIT      \
    do {    \
        if (unlikely(neta_logblock_checkhasinit())) {   \
            return -1;  \
        }   \
    } while(0)

#endif
