#ifndef NETA_LOGBLOCK_H
#define NETA_LOGBLOCK_H

#include "neta_lognode.h"

/*
* See the macro defines in file neta_lognode_ioctl_common.h.
*/

typedef struct neta_logblockgroup neta_logblockgroup;

typedef struct neta_logblockreference
{
    /*
    * Atomic add or minus operation.
    * The refcount is to decide who should put the log block.
    * The person who minus refcount to 1 need to delete the log block.
    * After delete the log block,
    * 0 means nobody refer it and the delete operation is done.
    * See refcount details on web https://hozonauto.feishu.cn/wiki/NwC4waZkjiQTTKkvWAIcEbh6nmh .
    * When need to add the refcount by consumer, you need to atomic_cmpxchg
    * successfully set the consumercount first.
    */
    u32                                 refcount;
    /*
    * Atomic exchange operation.
    * Log consumer count. The current strategy is that we let only one
    * thread use the log block simultaneously.
    * Currently, only value 0 or 1.
    */
    u32                                 consumercount;
} neta_logblockreference;

/*
* We use atomic_add_return to the expectedxxx variable here in the current
* file by change the pointer to atomic_t* and then use function atomic_add_return.
* We need to ensure sizeof atomic_t is equal to u32.
*/
STATIC_ASSERT(sizeof(atomic_t) == sizeof(u32));

/*
* It is an auxiliary structure to record the page info when allocage
* pages.
* The structure manages the block is of continuous physical memory
* of size NETA_BUDDY_MAXSIZE.
*/
typedef struct neta_logblockbuddy
{
    /*
    * The size of the page is certain, 4M bytes.
    * Use alloc_pages() to get the pointer.
    * Use free_pages() to free the resource.
    */
    struct page*                        ppage;
    /*
    * 1 means the page has been called SetPageReserved().
    * Use ClearPageReserved() to clear page reserved tag.
    * Not all log pages are reserved.
    * Currently, no log block group is reserved.
    * Later maybe only the first log block group(properly 128MB) is
    * reserved.
    */
    u32                                 breserved;
    // the base virtual address of the log block buddy
    void*                               vaddrbuddy;
    // the base physical address of the log block buddy
    u64                                 paddrbuddy;
    // block used count in the buddy, 0 - NETA_LOGBLOCKMAXCOUNT_PERBUDDY
    u32                                 usedcountinbuddy;
    /*
    * Use atomic_add_return to add it(once add 1).
    * Use & NETA_LOGBLOCKMAXCOUNT_PERBUDDY to get the buddy index.
    * For neta_logblock_tryget_in_buddy use.
    * Find vacant log block of the specific expectedblockindexofbuddy block buddy first.
    * When the expectedblockindexofbuddy is not vacant, logic will check every log block
    * in the buddy one by one until all of the block in the buddy has been checked.
    * And then if still cannot find vacant log block, logic will return with
    * the *o_pbget is 0.
    */
    u32                                 expectedblockindexofbuddy;
} neta_logblockbuddy;

/*
* Stored in the log block.
* Tag the correspondent neta_logblockbuddy and the offset in the entry.
*/
typedef struct neta_logblockbuddyentry
{
    neta_logblockbuddy*                 pbuddy;
    // the block index of the buddy,
    u32                                 blockindexofbuddy;
    // offset to the pbuddy base address.
    u32                                 offset;
} neta_logblockbuddyentry;

typedef struct neta_logblockgroupentry
{
    neta_logblockgroup*                 pgroup;
    // the block index of the group, 0 ~ NETA_LOGBLOCKMAXCOUNT_PERGROUP-1
    u32                                 blockindexofgroup;
} neta_logblockgroupentry;

typedef struct neta_logblockproducerthreadinfo
{
    struct mutex                        mutexthreadinfo;
    // 0 or 1, 0 means the logblock has been put back by the logblock producer
    u32                                 bproducing;
    // of logblock producer
    s32                                 processid;
    // of logblock producer
    s32                                 threadid;
    // of logblock producer
    u64                                 tscns;
    char                                name[LOGBLOCK_NAME_LENGTH];
} neta_logblockproducerthreadinfo;

/*
* The control structure of one specific log block.
* The log writer will only map the specific log block to
* user space. One new thread will map the logblock once.
*/
typedef struct neta_logblock
{
    // begin from 0, index of group
    u32                                 groupindex;
    // begin from 0, index of block in the group, equal to blockindexofgroup of neta_logblockgroupentry
    u32                                 blockindex;
    neta_logblockproducerthreadinfo     producerthreadinfo;
    /*
    * The kernel virtual base address of the current log block.
    */
    void*                               vaddrlogblock;
    /*
    * Point to one of the neta_logblockgroupshare's neta_logblockwroffset structure
    * array element.
    */
    neta_logblockwroffset*              pwroffset;
    /*
    * See note inside the structure.
    */
    neta_logblockreference              reference;
    neta_logblockgroupentry             groupentry;
    neta_logblockbuddyentry             buddyentry;
    /*
    * Default value is 0.
    * Valid when between rpalreadyhalf and wpalreadyhalf of neta_log structure.
    * You should check wether the correspondent logblock slot is the current logblock.
    */
    u32                                 alreadyhalfindex;
} neta_logblock;

/*
* We combine info of more than one log block to one variable, like we tag
* dirty bit every bit correspondent to one specific log block.
*/
typedef struct neta_logblockgroupcombine
{
    /*
    * Point to the neta_logblockgroupshare's neta_logblockgroupdirtycombine structure.
    */
    neta_logblockgroupdirtycombine*    pdirtycombine;
} neta_logblockgroupcombine;

/*
* The control structure of one specific log block group.
* Kernel mode code use only, the user mode code do not use it.
*/
typedef struct neta_logblockgroup
{
    neta_logblock                       parraylogblock[NETA_LOGBLOCKMAXCOUNT_PERGROUP];
    neta_logblockbuddy                  parraylogblockbuddy[NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP];
    // begin from 0
    u32                                 groupindex;
    // we record all of the combined data here in the neta_logblockgroupcombine structure
    neta_logblockgroupcombine           groupcombine;
    /*
    * Use atomic_add_return to add it(once add 1).
    * Use & NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP to get the buddy index.
    * For neta_logblock_tryget_in_group use.
    * Find vacant log block of the specific expectedbuddyindex block buddy first.
    * When cannot find vacant log block of the expectedbuddyindex, logic will check
    * every log block buddy one by one until all of the buddy in the group has been
    * checked. And then if still cannot find vacant log block, logic will return with
    * the *o_pbget is 0.
    * We
    */
    u32                                 expectedbuddyindex;
    // block used count in the group, 0 - NETA_LOGBLOCKMAXCOUNT_PERGROUP
    u32                                 usedcountingroup;
    // 1 means already create the new group, 0 means not
    u32                                 already_created;
} neta_logblockgroup;

/*
* When return 0, it mean successfully create logblockgroup, output to the
* *o_pgroup using the default i_breserved_default as page breserved setting.
* Return <0 means error.
* No any mutex lock inside.
* The function will not do neta_logblockgroup pointer link operation.
* The i_groupindex should always be the last log block group index, in other way, you
* should call the function to create log block group one by one.
* When return 0, *o_pppage is the page of the output neta_logblockgroup.
* When return 0, *o_pppagegroupshare is the page of the output neta_logblockgroupshare.
*/
s32 neta_logblockgroup_create(neta_logblockgroup** o_ppgroup, struct page** o_pppagegroup,
    neta_logblockgroupshare** o_ppgroupshare, struct page** o_pppagegroupshare,
    u32 i_groupindex, u32 i_breserved_default);
/*
* Current strategy: we only destroy the neta_logblockgroup when rmmod the ko.
* Delete the input log block group.
*/
s32 __attribute__((section("text.unlikely"))) neta_logblockgroup_destroy(neta_logblockgroup* i_pgroup, struct page* i_ppagegroup,
    neta_logblockgroupshare* i_pgroupshare, struct page* i_ppagegroupshare);

/*
* Mutex lock/unlock inside. Mutex for malloc new group.
* return 0 means we the input i_newgroupindex log block group can be used now.
* We call the function after we check all of the exist log block group and find out that
* all of them have no enough log block.
*/
s32 neta_logblockgroup_neednewgroup(u32 i_newgroupindex);

/*
* We first use neta_logblockgroup_create() to create a log block group and add it to
* the neta log node.
*/
s32 neta_logblockgroup_driverinit(void);

/*
* Current strategy: we only destroy the neta_logblockgroup when rmmod the ko.
* We call the neta_logblockgroup_destroy() to destroy the log block group one by one.
* There are no user operation any more when call the deinit function.
*/
s32 __attribute__((section("text.unlikely"))) neta_logblockgroup_driverdeinit(void);

/*
* *o_pbget is 0 or 1, 1 means get the logblock, 0 means not get, the log block buddy
* has no enough vacant log block to get.
* When *o_pbget is 1, *o_pplogblock is the vacant logblock just get, and the logblock
* has been occupied within the function.
* See note of expectedblockindexofbuddy in neta_logblockbuddy.
* Set i_trytimes to 0 to mean try every log block in the buddy. Otherwise, it will only
* check the specific try times.
*/
s32 neta_logblock_tryget_in_buddy(neta_logblock** o_pplogblock, u32 i_groupindex, u32 i_buddyindex, u32 i_trytimes, u32* o_pbget);
/*
* *o_pbget is 0 or 1, 1 means get the logblock, 0 means not get, the log block group
* has no enough vacant log block to get.
* When *o_pbget is 1, *o_pplogblock is the vacant logblock just get, and the logblock
* has been occupied within the function.
* See note of expectedbuddyindex in neta_logblockgroup.
*/
s32 neta_logblock_tryget_in_group(neta_logblock** o_pplogblock, u32 i_groupindex, u32* o_pbget);
/*
* Triggered by log producer only.
* Get a vacant log block place and output the pointer by *o_pplogblock.
* return 0 means no error.
* return other value means occur error.
*/
s32 neta_logblock_get(neta_logblock** o_pplogblock);
/*
* Triggerd by log producer or log consumer. Call the log producer or consumer specific record
* before call the neta_logblock_tryput function.
* *o_pbput is 0 or 1, 1 means put the logblock back, 0 means not put it back due to the log
* block is still tagging dirty.
* The refcount in struct neta_logblockreference minus to 2 triggers the function logic.
* When the refcount just minus to 2, we need to check whether the logblock is tagged dirty or
* not. When it is not tagged dirty, the function will call neta_logblock_put to put the log
* block back.
*/
s32 neta_logblock_tryput(neta_logblock* i_plogblock, u32* o_pbput);
/*
* Triggered by log producer or log consumer.
* Logic atomic minus refcount in struct neta_logblockreference, when minus to 1, it will trigger
* the log block put operation once. After all of the logblock operation is done, it will set the
* reference count back to 0.
* Tips: 1 means there is someone putting the log block back.
*/
s32 neta_logblock_put(neta_logblock* i_plogblock);

#if (NETA_DIRTYBIT_BY_BYTE == 1)
static inline s32 __always_inline neta_logblock_tagdirty(neta_logblockgroup* io_pgroup, u32 i_blockindex)
{
    volatile u8* pu8;
    NETA_DMB;
    pu8 = (volatile u8*)&io_pgroup->groupcombine.pdirtycombine->parrayu64dirtybit[0];
    pu8 += i_blockindex;
    *pu8 = 1;
    return 0;
}
static inline s32 __always_inline neta_logblock_tagnotdirty(neta_logblockgroup* io_pgroup, u32 i_blockindex)
{
    volatile u8* pu8;
    NETA_DMB;
    pu8 = (volatile u8*)&io_pgroup->groupcombine.pdirtycombine->parrayu64dirtybit[0];
    pu8 += i_blockindex;
    *pu8 = 0;
    return 0;
}
#else
static inline s32 __always_inline neta_logblock_tagdirty(neta_logblockgroup* io_pgroup, u32 i_blockindex)
{
    volatile u64* pu64;
    u64 tu64;
    NETA_DMB;
    pu64 = &io_pgroup->groupcombine.pdirtycombine->parrayu64dirtybit[i_blockindex >> 6];
    tu64 = *pu64;
    *pu64 = (tu64 | ((u64)1 << (i_blockindex & 0x3F)));
    return 0;
}
static inline s32 __always_inline neta_logblock_tagnotdirty(neta_logblockgroup* io_pgroup, u32 i_blockindex)
{
    volatile u64* pu64;
    u64 tu64;
    NETA_DMB;
    pu64 = &io_pgroup->groupcombine.pdirtycombine->parrayu64dirtybit[i_blockindex >> 6];
    tu64 = *pu64;
    *pu64 = (tu64 & (~((u64)1 << (i_blockindex & 0x3F))));
    return 0;
}
#endif

#if (NETA_DIRTYBIT_BY_BYTE == 1)
/*
* return 0 means not dirty
* return 1 means dirty
*/
static inline u32 __always_inline neta_logblock_returnbdirty(neta_logblockgroup* io_pgroup, u32 i_blockindex)
{
    u8* pu8;
    pu8 = (u8*)&io_pgroup->groupcombine.pdirtycombine->parrayu64dirtybit[0];
    pu8 += i_blockindex;
    return (u32)(*pu8);
}
#else
/*
* return 0 means not dirty
* return 1 means dirty
*/
static inline u32 __always_inline neta_logblock_returnbdirty(neta_logblockgroup* io_pgroup, u32 i_blockindex)
{
    u64 vu64 = io_pgroup->groupcombine.pdirtycombine->parrayu64dirtybit[i_blockindex >> 6];
    u32 offset = (i_blockindex & 0x3F);
    return ((U64BITH2L(vu64, offset, offset)) ? 1 : 0);
}
#endif

s32 neta_logblock_iteratecheckwprp(void);
s32 neta_logblock_alreadyhalf(neta_logblock* io_plogblock);

/*
* dma_buf part
*/

/*
* Get the dma_buf fd of log block.
*/
s32 neta_logblock_dmabufgetfd(neta_logblock* io_plogblock, s32* o_pdmabuffd);
/*
* Get the dma_buf fd of log block group share.
*/
s32 neta_logblockgroupcontrolinfo_dmabufgetfd(neta_lognodeprocessproducer* i_pprocessproducer, u32 i_groupindex, s32* o_pdmabuffd);
/*
* Get the dma_buf fd of the total log block group.
*/
s32 neta_logblockgroup_dmabufgetfd(neta_lognodeprocessconsumer* i_pprocessconsumer, u32 i_groupindex, s32* o_pdmabuffd);

#endif
