#ifndef NETA_LOGNODE_IOCTL_IMPL_H
#define NETA_LOGNODE_IOCTL_IMPL_H

#include "neta_lognode.h"

/*
* List all of the internal strategy macro here.
*/

/*
* Macro about the try get vacant log block strategy in the neta_logblock_tryget_in_group
* function.
*/
/*
* We try get several times before the step check whether the log block group has enough vacant log block.
*/
#define INTERNAL_STRATEGY_TRYGETINBUDDY_TIMES_BEFORE_CHECK_ENOUGH           4
/*
* We try get several log block in the buddy before the step check whether the log block group has enough
* vacant log block.
*/
#define INTERNAL_STRATEGY_TRYGETBUDDYCOUNT_TIMES_BEFORE_CHECK_ENOUGH        3
#if (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_128K)
/*
* When the current log block used count is bigger than the value, it means there are no enough vacant log
* block, need to create kernel thread to need new group.
*/
#define INTERNAL_STRATEGY_NEEDTHREADCREATEGROUP_IN_GROUP_USEDCOUNT          0x300
/*
* When the current log block used count is bigger than the value, it means there are no enough vacant log
* block.
*/
#define INTERNAL_STRATEGY_NOENOUGHVACANTLOGBLOCK_IN_GROUP_USEDCOUNT         0x380
/*
* You can change the 0x80 value, do not care about it.
*/
STATIC_ASSERT(INTERNAL_STRATEGY_NOENOUGHVACANTLOGBLOCK_IN_GROUP_USEDCOUNT < NETA_LOGBLOCKMAXCOUNT_PERGROUP - 0x40);
#elif (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K)
/*
* When the current log block used count is bigger than the value, it means there are no enough vacant log
* block, need to create kernel thread to need new group.
*/
#define INTERNAL_STRATEGY_NEEDTHREADCREATEGROUP_IN_GROUP_USEDCOUNT          0xC0
/*
* When the current log block used count is bigger than the value, it means there are no enough vacant log
* block.
*/
#define INTERNAL_STRATEGY_NOENOUGHVACANTLOGBLOCK_IN_GROUP_USEDCOUNT         0xE0
/*
* You can change the 0x10 value, do not care about it.
*/
STATIC_ASSERT(INTERNAL_STRATEGY_NOENOUGHVACANTLOGBLOCK_IN_GROUP_USEDCOUNT < NETA_LOGBLOCKMAXCOUNT_PERGROUP - 0x10);
#elif (NETA_LOGBLOCK_USING == NETA_LOGBLOCK_USING_512K_DOUBLE)
/*
* When the current log block used count is bigger than the value, it means there are no enough vacant log
* block, need to create kernel thread to need new group.
*/
#define INTERNAL_STRATEGY_NEEDTHREADCREATEGROUP_IN_GROUP_USEDCOUNT          0x180
/*
* When the current log block used count is bigger than the value, it means there are no enough vacant log
* block.
*/
#define INTERNAL_STRATEGY_NOENOUGHVACANTLOGBLOCK_IN_GROUP_USEDCOUNT         0x1C0
/*
* You can change the 0x20 value, do not care about it.
*/
STATIC_ASSERT(INTERNAL_STRATEGY_NOENOUGHVACANTLOGBLOCK_IN_GROUP_USEDCOUNT < NETA_LOGBLOCKMAXCOUNT_PERGROUP - 0x20);
#endif

#define INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CHECK_GROUP_INDEX              3
#define INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CLEAR_GROUP_INDEX_BEGIN        1
#define INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CLEAR_GROUP_INDEX_END          3
STATIC_ASSERT(INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CLEAR_GROUP_INDEX_END < NETA_LOGBLOCKGROUP_MAXCOUNT);
#define INTERNAL_STRATEGY_ITERATEGETBLOCK_TRYTIMES                          3

/*
* We try get several times after we find that there are enough vacant log block.
*/
#define INTERNAL_STRATEGY_TRYGETINBUDDY_TIMES_AFTER_CHECK_ENOUGH            NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP
/*
* We try get several log block in the buddy after the step check wether the log block group has enough
* vacant log block.
*/
#define INTERNAL_STRATEGY_TRYGETBUDDYCOUNT_TIMES_AFTER_CHECK_ENOUGH         NETA_LOGBLOCKMAXCOUNT_PERBUDDY


/*
* List all of the function of xx_ioctl_xx.h defined structure.
*/

/*
* Log block producer.
*/

s32 internal_neta_dev_logblocknode_ioctl_getlogblock(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logwriterioctl_getlogblock* io_pgetlogblock);
s32 internal_neta_dev_logblocknode_ioctl_putlogblock(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logwriterioctl_putlogblock* io_pputlogblock);
s32 internal_neta_dev_logblocknode_ioctl_tagdirty(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logwriterioctl_tagdirty* io_ptagdirty);

/*
* Log block consumer.
*/

s32 internal_neta_dev_logblocknode_ioctl_getnextlogblocktoread(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logreaderioctl_getnextlogblocktoread* io_pgetnextlogblocktoread);
s32 internal_neta_dev_logblocknode_ioctl_finishlogblockread(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logreaderioctl_finishlogblockread* io_pfinishlogblockread);
s32 internal_neta_dev_logblocknode_ioctl_readerthreadquit(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logreaderioctl_readerthreadquit* io_preaderthreadquit);
s32 internal_neta_dev_logblocknode_ioctl_wakeupreaderthread(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logreaderioctl_wakeupreaderthread* io_pwakeupreaderthread);
s32 internal_neta_dev_logblocknode_ioctl_checklogblockproducerthread(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logreaderioctl_checklogblockproducerthread* io_pchecklogblockproducerthread);
s32 internal_neta_dev_logblocknode_ioctl_getglobalstatus(neta_lognodeprocessuser* io_pprocessuser, 
    neta_logreaderioctl_getglobalstatus* io_pgetglobalstatus);

/*
* You must ensure the input i_start value within 0~63.
* return value: 0~64, when 64 means cannot find any bit value 1.
* Return the smallest bit index of value 1.
* The return value should be bigger than or equal to i_start.
*/
u32 internal_neta_returnsmallestbitindex(u64 i_u64, u32 i_start);
/*
* return 0 or 1.
* return 1 only when meet all of the following condition:
* 1) no other consumer is using the log block 
* 2) the read point is not equal to the write point
* When return 1, the function will add the consumer reference and the 
* log block total reference.
*/
u32 internal_neta_returnbneedtoread(neta_logblock* io_plogblock, u32* o_proffset_begin, u32* o_proffset_end);
/*
* return 0 or 1.
* When return 0, it means find a log block part to read, and you need to 
* check *o_pbitindextochange, when *o_pbitindextochange is 64, it means 
* you should add the currcombineindex.
*/
u32 internal_neta_returnbneedfindnextcombine(u32 i_groupindex, neta_logblockgroupdirtycombine* i_pcombine, 
    u32 i_currcombineindex, u32 i_currbitindex, u32* io_pbhasonetagdirty, neta_logblock** o_pplogblock,
    u32* o_proffset_begin, u32* o_proffset_end, u32* o_pbitindextochange);

/*
* Must be called by the process of the input io_pthreadconsumer.
* Called by internal_neta_dev_logblocknode_ioctl_readerthreadquit and 
* the release operation triggered by process quit.
* i_plogdesc is for log, should be .rodata
*/
s32 internal_neta_logblocknode_readerthreadquit(neta_lognodethreadconsumer* io_pthreadconsumer, 
    const char* i_plogdesc);

#endif
