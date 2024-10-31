#ifndef NETA_LOGNODE_IOCTL_READER_H
#define NETA_LOGNODE_IOCTL_READER_H

#include "neta_lognode_ioctl_writer.h"

/*
* See the ioctl basic info in neta_lognode_ioctl_writer.h.
*/

/*
* Every log reader may handle more than one log block group.
*/
typedef struct neta_logreaderioctl_getnextlogblocktoread {
    /*
    * The cookie correspondent to the reader thread, which is being used by 
    * driver. 0 is the default value, set by driver, and need to set the value 
    * every time you do ioctl.
    * Do not put processid and threadid and tsc here to cut down ioctl size.
    */
    u64 io_readerthread_cookie;
    // 0 means do not wait, -1 means wait inifinitely
    u32 i_timeoutms;

    /*
    * The global status part. The value is always valid.
    */
    u32 o_alreadyhalfcounter;
    
    /*
    * -1 or valid group index
    * -1 and o_premapgroupindex is -1: not get any log block to read within the timeout period
    * -1 and o_premapgroupindex is not -1: need to pre-map the log block groupfd.
    * other: get log block to read, begin from 0, of log block group
    */
    u32 o_groupindex;
    /*
    * Valid only when o_groupindex is -1.
    * -1 or valid group index.
    * When it is valid group index, you need to pre-map the log block groupfd of the valid
    * group index.
    */
    u32 o_premapgroupindex;
    
    /*
    * The following members are all invalid when o_groupindex is -1 and o_premapgroupindex
    * is -1.
    * When o_groupindex is -1, but o_premapgroupindex is positive, only o_logblockgroupfd 
    * is valid meaning that you need to pre-map the log block groupfd. The per-map ioctl 
    * will be called only once per process.
    * The o_premapgroupindex is only valid when o_groupindex is -1.
    */
    
    /*
    * Begin from 0, of log block inside the specific log block 
    * group. The log block to be read later.
    */
    u32 o_blockindex;
    // tag the logblock slot in the internal array
    u64 o_logblock_cookie;
    // the process id of the log block producer
    s32 o_producer_processid;
    // the process id of the log block producer
    s32 o_producer_threadid;
    // the tsc ns time when the log block producer open log block handle
    u64 o_producer_tscns;
    char o_producer_name[LOGBLOCK_NAME_LENGTH];
    u32 o_roffset_begin;
    u32 o_roffset_end;

    /*
    * You need to check whether you has already map the whole log block group of 
    * the specific group. You should store the mapped log block group by the group
    * index.
    */

    /*
    * Size is constant NETA_LOGBLOCKGROUP_SIZE.
    */
    s32 o_logblockgroupfd;

    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logreaderioctl_getnextlogblocktoread;

/*
* You should ensure only the i_readerthread_cookie thread itself can call the ioctl.
*/
typedef struct neta_logreaderioctl_finishlogblockread {
    /*
    * The cookie correspondent to the reader thread, which is being used by 
    * driver. Not 0.
    */
    u64 i_readerthread_cookie;

    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logreaderioctl_finishlogblockread;

typedef struct neta_logreaderioctl_readerthreadquit {
    /*
    * The cookie correspondent to the reader thread, which is being used by 
    * driver. Not 0.
    */
    u64 i_readerthread_cookie;

    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logreaderioctl_readerthreadquit;

typedef struct neta_logreaderioctl_wakeupreaderthread {
    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logreaderioctl_wakeupreaderthread;

/*
* The member of the structure is get by GetNextLogBlockToRead.
*/
typedef struct neta_logreaderioctl_checklogblockproducerthread {
    /*
    * Tag the logblock slot in the internal array.
    */
    u64 i_logblock_cookie;
    s32 i_process_id;
    s32 i_thread_id;
    /*
    * The tsc ns time when the log block producer thread open log block handle.
    */
    u64 i_tscns;

    /*
    * 0 or 1, 1 means the producer thread may still be alive
    * 0 means the producer thread is not alive now
    */
    u32 o_bproducing;

    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logreaderioctl_checklogblockproducerthread;

/*
* The global status.
*/
typedef struct neta_logreaderioctl_getglobalstatus {
    /*
    * The counter of the total already half counter, count from the beginning.
    * May overflow, you need to minus the origin value last get to check whether
    * the total system is busy or not.
    */
    u32 o_alreadyhalfcounter;
    // 0 means no error, negative value means error
    s32 o_ret;
} neta_logreaderioctl_getglobalstatus;

/*
* Get next log block to read. One log block is being read by only one thread.
*/
#define NETA_LOGNODE_IO_LOGREADER_GETNEXTLOGBLOCKTOREAD \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGREADER_BASE + 1, neta_logreaderioctl_getnextlogblocktoread)
/*
* Finish log block to read of specific thread.
*/
#define NETA_LOGNODE_IO_LOGREADER_FINISHLOGBLOCKREAD \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGREADER_BASE + 2, neta_logreaderioctl_finishlogblockread)
/*
* The reader thread of the i_readerthread_cookie has quit.
*/
#define NETA_LOGNODE_IO_LOGREADER_READERTHREADQUIT \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGREADER_BASE + 3, neta_logreaderioctl_readerthreadquit)
/*
* Wake up one or several reader thread immediately.
*/
#define NETA_LOGNODE_IO_LOGREADER_WAKEUPREADERTHREAD \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGREADER_BASE + 4, neta_logreaderioctl_wakeupreaderthread)
/*
* Check whether the specific logblock producer thread is alive or not.
*/
#define NETA_LOGNODE_IO_LOGREADER_CHECKLOGBLOCKPRODUCERTHREAD \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGREADER_BASE + 5, neta_logreaderioctl_checklogblockproducerthread)
/*
* Get global status.
*/
#define NETA_LOGNODE_IO_LOGREADER_GETGLOBALSTATUS \
    _IOWR(NETA_LOGNODE_TYPEMAGIC, NETA_LOGNODE_IO_LOGREADER_BASE + 6, neta_logreaderioctl_getglobalstatus)

#endif
