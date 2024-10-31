#ifndef NETA_LOGNODE_IOCTL_COMMON_H
#define NETA_LOGNODE_IOCTL_COMMON_H

#include "logblockreader.h"



/*
* We define the macro which the user mode should also use here. 
*/

// currently, only support limited count log block group
#define NETA_LOGBLOCKGROUP_MAXCOUNT             4

#define NETA_LOGBLOCK_USING_128K                1
#define NETA_LOGBLOCK_USING_512K                2
#define NETA_LOGBLOCK_USING_512K_DOUBLE         3
#define NETA_LOGBLOCK_USING_2M                  4
#define NETA_LOGBLOCK_USING_4M                  5

#define NETA_LOGBLOCK_USING                     NETA_LOGBLOCK_USING_512K_DOUBLE

#if (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K_DOUBLE)
// 256MB bytes of one log block group
#define NETA_LOGBLOCKGROUP_SIZE                 0x10000000
#else
// 128MB bytes of one log block group
#define NETA_LOGBLOCKGROUP_SIZE                 0x8000000
#endif

#if (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_128K)
// 128k bytes
#define NETA_LOGBLOCK_SIZE                      0x20000
// 1024 log block per log block group
#define NETA_LOGBLOCKMAXCOUNT_PERGROUP          0x400
#define NETA_LOGBLOCKMAXCOUNT_PERBUDDY          32
#define NETA_LOGBLOCKMAXCOUNT_PERBUDDY_POWER    5
#define NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP     32
#define NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP_POWER   5
#elif (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K)
// 128k bytes
#define NETA_LOGBLOCK_SIZE                      0x80000
// 256 log block per log block group
#define NETA_LOGBLOCKMAXCOUNT_PERGROUP          0x100
#define NETA_LOGBLOCKMAXCOUNT_PERBUDDY          8
#define NETA_LOGBLOCKMAXCOUNT_PERBUDDY_POWER    3
#define NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP     32
#define NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP_POWER   5
#elif (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K_DOUBLE)
// 128k bytes
#define NETA_LOGBLOCK_SIZE                      0x80000
// 512 log block per log block group
#define NETA_LOGBLOCKMAXCOUNT_PERGROUP          0x200
#define NETA_LOGBLOCKMAXCOUNT_PERBUDDY          8
#define NETA_LOGBLOCKMAXCOUNT_PERBUDDY_POWER    3
#define NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP     64
#define NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP_POWER   6
#endif

STATIC_ASSERT(NETA_LOGBLOCK_SIZE * NETA_LOGBLOCKMAXCOUNT_PERGROUP 
    == NETA_LOGBLOCKGROUP_SIZE);
STATIC_ASSERT(NETA_LOGBLOCK_SIZE * NETA_LOGBLOCKMAXCOUNT_PERBUDDY 
    == NETA_BUDDY_MAXSIZE);
STATIC_ASSERT(NETA_LOGBLOCKMAXCOUNT_PERBUDDY * NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP 
    == NETA_LOGBLOCKMAXCOUNT_PERGROUP);

// set it 0 or 1, set it to 1 to mean use one byte as the dirty bit
#define NETA_DIRTYBIT_BY_BYTE           1

#if (NETA_DIRTYBIT_BY_BYTE == 1)

// only support 512k mode dirtybit by byte
STATIC_ASSERT(NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K || NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K_DOUBLE);

#if (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K)
#define NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT 32
#elif (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K_DOUBLE)
#define NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT 64
#endif

STATIC_ASSERT(NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT * 8 
    == NETA_LOGBLOCKMAXCOUNT_PERGROUP);

#else
#if (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_128K)
// log block combine part
/*
* Every log block correspondent to one dirty bit.
* In one log block group, there are 
* NETA_LOGBLOCKMAXCOUNT_PERGROUP/NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT 
* u64 elements.
*/
#define NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT 16
#elif (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K)
// log block combine part
/*
* Every log block correspondent to one dirty bit.
* In one log block group, there are 
* NETA_LOGBLOCKMAXCOUNT_PERGROUP/NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT 
* u64 elements.
*/
#define NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT 4
#else
#endif

STATIC_ASSERT(NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT * 64 
    == NETA_LOGBLOCKMAXCOUNT_PERGROUP);

#endif



/*
* Other internal strategy macro.
* The count is of the already half buffer array.
* It is not about the NETA_LOGBLOCK_USING mode.
*/
#define NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT            32
/*
* 0 or 1
* 1 means enable alreadyhalf get strategy.
*/
#define NETA_LOGBLOCK_ALREADYHALF_STRATEGY_ENABLE                   1
/*
* 0 or 1
* 1 means enable simple already half logic
*/
#define NETA_LOGBLOCK_ALREADYHALF_SIMPLE_LOGIC                      0
/*
* 0 or 1
* 1 means enable alreadyhalf debug log.
*/
#define NETA_LOGBLOCK_ALREADYHALF_DEBUG_LOG                         0
/*
* 0 or 1
* 1 means enable slow path logblock get when group count exceeds.
*/
#define NETA_LOGBLOCK_GET_SLOW_PATH_ENABLE                          1

typedef struct neta_logblockgroupdirtycombine
{
    // Every log block correspondent to one dirty bit.
    volatile u64                    parrayu64dirtybit[NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT];
} neta_logblockgroupdirtycombine;

/*
* Log block write offset and reade offset.
*/
typedef struct neta_logblockwroffset
{
    // see note in LogBlockWriterInfo's pwoffset
    volatile u32                     woffset;
    // see note in LogBlockReaderInfo's roffset_begin and roffset_end
    volatile u32                     roffset;
} neta_logblockwroffset;

/*
* The log reader will map the total neta_logblockgroupshare structure to 
* user space.
*/
typedef struct neta_logblockgroupshare
{
    neta_logblockgroupdirtycombine     dirtycombine;
    neta_logblockwroffset              parraywroffset[NETA_LOGBLOCKMAXCOUNT_PERGROUP];
} neta_logblockgroupshare;

#endif
