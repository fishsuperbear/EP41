#ifndef LOGBLOCKWRITER_H
#define LOGBLOCKWRITER_H

#include "neta_node_common.h"

typedef void*   LogBlockHandle;

typedef enum LOGBLOCK_NOTIFYTYPE
{
    LOGBLOCK_NOTIFYTYPE_NEARFULL = 0,
    LOGBLOCK_NOTIFYTYPE_SYSDUMP = 1,
} LOGBLOCK_NOTIFYTYPE;

typedef struct LogBlockWriterInfo
{
    void*               vaddr;  // the beginning virtual address of the log block
    u32                 blocksize;  // byte count, always 2 power, bigger than 4k
    /*
    * The pointer to the write offset, you set the value like *pwoffset = *pwoffset + xx;
    * You should do nothing when the *pwoffset is bigger than the blocksize.
    */
    volatile u32*       pwoffset;
    /*
    * The pointer to the read offset, you can only read the offset value by *proffset.
    * You cannot change the *proffset value.
    */
    volatile u32*       proffset;
} LogBlockWriterInfo;

// the max length of i_pname is LOGBLOCK_NAME_LENGTH (including '\0') defined in logblockreader.h
s32 OpenLogBlockHandle(const char* i_pname, LogBlockHandle* o_plogblockhandle);
s32 GetLogBlockWriterInfo(LogBlockHandle i_logblockhandle, LogBlockWriterInfo* o_plogblockinfo);
s32 TagLogBlockDirty(LogBlockHandle i_logblockhandle);
s32 LogBlockSendNotify(LogBlockHandle i_logblockhandle, LOGBLOCK_NOTIFYTYPE i_notifytype);

#endif
