#ifndef LOGBLOCKWRITER_H
#define LOGBLOCKWRITER_H

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

typedef void*   LogBlockHandle;

enum LOGBLOCK_NOTIFYTYPE
{
    LOGBLOCK_NOTIFYTYPE_NEARFULL = 0,
    LOGBLOCK_NOTIFYTYPE_SYSDUMP = 1,
};

typedef struct LogBlockWriterInfo
{
    void*       vaddr;  // the beginning virtual address of the log block
    u32         blocksize;  // byte count, always 2 power, bigger than 4k
    /*
    * The pointer to the write offset, you set the value like **pwoffset = **pwoffset + xx;
    * You should do nothing when the **pwoffset is bigger than the blocksize.
    */
    u32*        pwoffset;
    /*
    * The pointer to the read offset, you can only read the offset value by *proffset.
    * You cannot change the *proffset value.
    */
    u32*        proffset;
} LogBlockWriterInfo;

s32 OpenLogBlockHandle(LogBlockHandle* o_plogblockhandle);
s32 GetLogBlockWriterInfo(LogBlockHandle i_logblockhandle, LogBlockWriterInfo* i_plogblockinfo);
s32 TagLogBlockDirty(LogBlockHandle i_logblockhandle);
s32 CloseLogBlockHandle(LogBlockHandle i_logblockhandle);
s32 ForceCloseLogBlockHandle(LogBlockHandle i_logblockhandle);
s32 LogBlockSendNotify(LogBlockHandle i_logblockhandle, LOGBLOCK_NOTIFYTYPE i_notifytype);

#ifdef __cplusplus
}
#endif

#endif
