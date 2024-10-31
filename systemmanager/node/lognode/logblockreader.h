#ifndef LOGBLOCKREADER_H
#define LOGBLOCKREADER_H

#include "neta_node_common.h"

typedef void*   LogBlockReaderThreadHandle;

typedef enum LOGBLOCK_READRETSTATUS
{
    LOGBLOCK_READRETSTATUS_GET = 0,
    LOGBLOCK_READRETSTATUS_TIMEOUT = 1,
} LOGBLOCK_READRETSTATUS;

#define LOGBLOCK_NAME_LENGTH        32

typedef struct LogBlockThreadInfo
{
    // tag the logblock slot in the internal array
    u64         logblock_cookie;
    s32         process_id;
    s32         thread_id;
    /*
    * The tsc ns time when the log block producer thread open log block handle.
    */
    u64         tscns;
    char        name[LOGBLOCK_NAME_LENGTH];
} LogBlockThreadInfo;

typedef struct LogBlockReaderInfo
{
    void*       vaddr;  // the beginning virtual address of the log block
    u32         blocksize;  // byte count, always 2 power, bigger than 4k
    /*
    * The offset is properly bigger than the blocksize, you should calculate 
    * the correct offset by yourself like the sentence below:
    * offset_begin = roffset_begin % blocksize;
    */
    u32         roffset_begin;
    /*
    * The offset is properly bigger than the blocksize, you should calculate 
    * the correct offset by yourself like the sentence below:
    * offset_end = roffset_end % blocksize; 
    */
    u32         roffset_end;
    /*
    * The log block producer thread info.
    */
    struct LogBlockThreadInfo producerthreadinfo;
} LogBlockReaderInfo;

s32 OpenLogBlockReaderThreadHandle(LogBlockReaderThreadHandle* o_preaderthreadhandle);
/*
* i_timeoutms: 0 means do not wait, -1 means wait infinitely
*/
s32 GetNextLogBlockToRead(LogBlockReaderThreadHandle i_readerthreadhandle, LogBlockReaderInfo* o_plogblockinfo, u32 i_timeoutms, LOGBLOCK_READRETSTATUS* o_preadretstatus);
s32 FinishLogBlockRead(LogBlockReaderThreadHandle i_readerthreadhandle);

/*
* Do not need to call OpenLogBlockReaderThreadHandle before calling the function.
* The logblock reader thread may set timeout to a large number, you can use the function to 
* immediately wake up one or several reader thread.
*/
s32 WakeUpReaderThread(void);
/*
* Do not need to call OpenLogBlockReaderThreadHandle before calling the function.
* Check whether the input producer thread is still producing log.
* *o_pbproducing is 0 or 1, when it is 1, it does not mean there are new log to read, it just 
* mean the producer thread may still be alive.
* When *o_pbproducing is 0, it means the producer thread is not alive now.
*/
s32 CheckLogBlockProducerThread(LogBlockThreadInfo i_producerthreadinfo, u32* o_pbproducing);

#endif
