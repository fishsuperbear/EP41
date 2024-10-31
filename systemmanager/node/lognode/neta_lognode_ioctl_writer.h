#ifndef NETA_LOGNODE_IOCTL_WRITER_H
#define NETA_LOGNODE_IOCTL_WRITER_H

#include "neta_lognode_ioctl_common.h"

#define NETA_LOGNODE_DEVPATH                    "/dev/netalog"
#define NETA_LOGNODE_TYPEMAGIC                  0x29
#define NETA_LOGNODE_IO_LOGWRITER_BASE          0x10
#define NETA_LOGNODE_IO_LOGREADER_BASE          0x20

/*
* Every log writer write one log block.
*/
typedef struct neta_logwriterioctl_getlogblock {
    // process id, get by getpid()
    s32 i_processid;
    // thread id, get by gettid()
    s32 i_threadid;
    // the time(ns unit) when do get log block ioctl, get by hw_plat_get_tsc_ns.
    u64 i_tscns;
    char i_name[LOGBLOCK_NAME_LENGTH];
    /*
    * The cookie correspondent to the log block thread, which is being used by 
    * driver. 0 is the default value, set by driver, and need to set the value 
    * every time you do ioctl.
    */
    u64 io_writerthread_cookie;
    // begin from 0, of log block group
    u32 o_groupindex;
    /*
    * Begin from 0, of log block inside the specific log block 
    * group.
    */
    u32 o_blockindex;
    /*
    * Always valid. Need use it to map. 
    * Size is constant NETA_LOGBLOCK_SIZE.
    */
    s32 o_logblockfd;

    /*
    * You need to check whether you has already map the control info of the specific 
    * group. You should store the control info by the group index.
    */

    /*
    * Size is not constant. Size is controlinfosize.
    */
    s32 o_controlinfofd;
    // byte count of control info, always valid.
    u32 o_controlinfosize;

    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logwriterioctl_getlogblock;

typedef struct neta_logwriterioctl_putlogblock {
    // process id, get by getpid()
    s32 i_processid;
    // thread id, get by gettid()
    s32 i_threadid;
    // the time(ns unit) when do get log block ioctl, get by hw_plat_get_tsc_ns.
    u64 i_tscns;
    /*
    * The cookie correspondent to the writer thread, which is being used by 
    * driver. Not 0.
    */
    u64 i_writerthread_cookie;
    // begin from 0, of log block group
    u32 i_groupindex;
    /*
    * Begin from 0, of log block inside the specific log block 
    * group.
    */
    u32 i_blockindex;
    
    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logwriterioctl_putlogblock;

/*
* TagLogBlockDirty trigger the ioctl, but not every time it is triggered.
* It will wake up all of the logblock consumer threads if they are sleeping.
*/
typedef struct neta_logwriterioctl_tagdirty {
    // -1 means there is no logblock need to immediately get
    u32 i_groupindex;
    u32 i_blockindex;
    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logwriterioctl_tagdirty;

/*
* Get a vacant log block to use. Get vacant log block of the current thread.
*/
#define NETA_LOGNODE_IO_LOGWRITER_GETLOGBLOCK \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGWRITER_BASE + 1, neta_logwriterioctl_getlogblock)
/*
* Put the specific log block back. Put the log block back of the current thread.
*/
#define NETA_LOGNODE_IO_LOGWRITER_PUTLOGBLOCK \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGWRITER_BASE + 2, neta_logwriterioctl_putlogblock)
/*
* Tag dirty. Not specific log block. We put it as logwriter part, but it can be 
* called in any context.
*/
#define NETA_LOGNODE_IO_LOGWRITER_TAGDIRTY \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGWRITER_BASE + 3, neta_logwriterioctl_tagdirty)

#endif
