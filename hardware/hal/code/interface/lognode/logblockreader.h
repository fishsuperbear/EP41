#ifndef LOGBLOCKREADER_H
#define LOGBLOCKREADER_H

#ifdef __cplusplus
extern "C" {
#endif

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
#define STATIC_ASSERT(truecond)     static_assert (truecond, "error")
#endif

STATIC_ASSERT(sizeof(void*) == 8);

STATIC_ASSERT(sizeof(u8) == 1);
STATIC_ASSERT(sizeof(u16) == 2);
STATIC_ASSERT(sizeof(u32) == 4);
STATIC_ASSERT(sizeof(u64) == 8);

enum LOGBLOCK_READRETSTATUS
{
    LOGBLOCK_READRETSTATUS_GET = 0,
    LOGBLOCK_READRETSTATUS_TIMEOUT = 1,
};

typedef struct LogBlockThreadInfo
{
    u32         process_id;
    u32         thread_id;
} LogBlockThreadInfo;

typedef struct LogBlockReaderInfo
{
    struct LogBlockThreadInfo   threadinfo;
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
} LogBlockReaderInfo;

/*
* i_timeoutms: 0 means do not wait, -1 means wait infinitely
*/
s32 GetNextLogBlockToRead(LogBlockReaderInfo* i_plogblockinfo, u32 i_timeoutms, LOGBLOCK_READRETSTATUS *o_preadretstatus);
s32 FinishLogBlockRead();
s32 LogBlockCleanUp();

#ifdef __cplusplus
}
#endif

#endif
