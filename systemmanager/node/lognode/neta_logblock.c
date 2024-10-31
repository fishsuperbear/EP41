#include "neta_log.h"


s32 neta_logblockgroup_create(neta_logblockgroup** o_pgroup, struct page** o_pppagegroup,
    neta_logblockgroupshare** o_ppgroupshare, struct page** o_pppagegroupshare,
    u32 i_groupindex, u32 i_breserved_default)
{
    s32 ret = 0;
    u32 pagei;
    u32 orderblockgroup;
    struct page* ppagegroup;
    u32 buddyi, blockiperbuddy, blocki;
    neta_logblockbuddy* pbuddy;
    neta_logblock* pblock;
    struct page* ppagebuddy;
    neta_logblockgroupentry groupentry;
    neta_logblockbuddyentry buddyentry;
    u32 ordergroupshare;
    struct page* ppagegroupshare;
    neta_logblockgroupshare* pgroupshare;
    u32 u64dirtyi;
    neta_logblockgroup* pgroup;
    if (i_breserved_default != 0)
    {
        ret = NETA_KERR_LOGBLOCKBUDDY_BRESERVED_1_NOT_SUPPORT;
        NETA_KLOG_ERR("i_breserved_default=%u, unexpected, should be 0!\n", i_breserved_default);
        return ret;
    }
    orderblockgroup = get_order(sizeof(neta_logblockgroup));
    NETA_KLOG_INFO("log block group create index[%u][size:0x%x,order:%u]\n", i_groupindex,
        (u32)(sizeof(neta_logblockgroup)), orderblockgroup);
    ppagegroup = alloc_pages(GFP_KERNEL, orderblockgroup);
    CHK_PTR_AND_RET_S32(ppagegroup, "alloc_pages blockgroup");
    for (pagei = 0; pagei < (1 << orderblockgroup); pagei++) {
        SetPageReserved(ppagegroup + pagei);
    }
    pgroup = (neta_logblockgroup*)page_address(ppagegroup);
    /*
    * Init parraylogblock of neta_logblockgroup.
    */
    ppagebuddy = NULL;
    groupentry.pgroup = pgroup;
    /*
    * Init group share.
    */
    ordergroupshare = get_order(sizeof(neta_logblockgroupshare));
    NETA_KLOG_INFO("log block group share create index[%u][size:0x%x[0x%x],order:%u]\n", i_groupindex,
        (u32)sizeof(neta_logblockgroupshare), (u32)(PAGE_SIZE * (1 << ordergroupshare)), ordergroupshare);
    ppagegroupshare = alloc_pages(GFP_KERNEL, ordergroupshare);
    CHK_PTR_AND_RET_S32(ppagegroupshare, "alloc_pages blockgroupshare");
    for (pagei = 0; pagei < (1 << ordergroupshare); pagei++) {
        SetPageReserved(ppagegroupshare + pagei);
    }
    pgroupshare = (neta_logblockgroupshare*)page_address(ppagegroupshare);
    /*
    * We need to init log block group share first before init log block.
    */
    for (u64dirtyi = 0; u64dirtyi < NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT; u64dirtyi ++)
    {
        pgroupshare->dirtycombine.parrayu64dirtybit[u64dirtyi] = 0;
    }
    for (blocki = 0; blocki < NETA_LOGBLOCKMAXCOUNT_PERGROUP; blocki ++)
    {
        pgroupshare->parraywroffset[blocki].woffset = 0;
        pgroupshare->parraywroffset[blocki].roffset = 0;
    }
    /*
    * Init log block buddy and then init every log block of the buddy inside the loop.
    */
    for (buddyi = 0; buddyi < NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP; buddyi ++)
    {
        pbuddy = &pgroup->parraylogblockbuddy[buddyi];
        ppagebuddy = alloc_pages(GFP_KERNEL, 10);   // 2^10 1024 * 4k = 4MB
        CHK_PTR_AND_RET_S32(ppagebuddy, "alloc_pages 4M");
        for (pagei = 0; pagei < (1 << 10); pagei++) {
            SetPageReserved(ppagebuddy + pagei);
        }
        pbuddy->ppage = ppagebuddy;
        pbuddy->breserved = 0;  // currently only support i_breserved_default 0
        pbuddy->vaddrbuddy = page_address(ppagebuddy);
        pbuddy->paddrbuddy = (u64)page_to_phys(ppagebuddy);
        pbuddy->usedcountinbuddy = 0;   // default 0 log block is used in the buddy
        pbuddy->expectedblockindexofbuddy = 0;  // expect the index 0 as the next vacant log block
        /*
        * Prepare the buddy entry to be set to log block.
        */
        buddyentry.pbuddy = pbuddy;
        for (blockiperbuddy = 0; blockiperbuddy < NETA_LOGBLOCKMAXCOUNT_PERBUDDY; blockiperbuddy ++)
        {
            blocki = buddyi*NETA_LOGBLOCKMAXCOUNT_PERBUDDY + blockiperbuddy;
            pblock = &pgroup->parraylogblock[blocki];
            buddyentry.blockindexofbuddy = blockiperbuddy;
            // every log block is NETA_LOGBLOCK_SIZE
            buddyentry.offset = blockiperbuddy * NETA_LOGBLOCK_SIZE;
            pblock->groupindex = i_groupindex;
            pblock->blockindex = blocki;
            mutex_init(&pblock->producerthreadinfo.mutexthreadinfo);
            pblock->vaddrlogblock = (void*)((u64)pbuddy->vaddrbuddy + (u64)buddyentry.offset);
            pblock->pwroffset = &pgroupshare->parraywroffset[blocki];
            pblock->reference.refcount = 0;
            pblock->reference.consumercount = 0;
            groupentry.blockindexofgroup = blocki;
            pblock->groupentry = groupentry;
            pblock->buddyentry = buddyentry;
            pblock->alreadyhalfindex = 0;
        }
    }
    /*
    * Init other member of neta_logblockgroup.
    */
    pgroup->groupindex = i_groupindex;
    pgroup->groupcombine.pdirtycombine = &pgroupshare->dirtycombine;
    pgroup->expectedbuddyindex = 0;
    pgroup->usedcountingroup = 0;   // default 0 log block is used in the group
    pgroup->already_created = 0;
    /*
    * Output pointers in the end.
    */
    *o_pgroup = pgroup;
    *o_pppagegroup = ppagegroup;
    *o_ppgroupshare = pgroupshare;
    *o_pppagegroupshare = ppagegroupshare;
    return 0;
}

s32 neta_logblockgroup_destroy(neta_logblockgroup* i_pgroup, struct page* i_ppagegroup,
    neta_logblockgroupshare* i_pgroupshare, struct page* i_ppagegroupshare)
{
    /*
    * Delete the log block inside the log block group first, then the i_pgroupshare and then
    * the log block group self.
    */
    u32 buddyi;
    neta_logblockbuddy* pbuddy;
    u32 blocki;
    u32 pagei;
    neta_logblock* plogblock;
    u32 orderblockgroup = get_order(sizeof(neta_logblockgroup));
    u32 ordergroupshare = get_order(sizeof(neta_logblockgroupshare));
    for (buddyi = 0; buddyi < NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP; buddyi ++)
    {
        pbuddy = &i_pgroup->parraylogblockbuddy[buddyi];
        if (pbuddy->ppage) {
            // log block
            for (pagei = 0; pagei < (1 << 10); pagei++) {
                ClearPageReserved(pbuddy->ppage + pagei);
            }
            free_pages((u64)pbuddy->vaddrbuddy, 10);
        }
    }
    if (i_pgroupshare) {
        for (pagei = 0; pagei < (1 << ordergroupshare); pagei++) {
            ClearPageReserved(i_ppagegroupshare + pagei);
        }
        free_pages((u64)i_pgroupshare, ordergroupshare);
    }
    /*
    * Destroy the logblock producer info mutex.
    */
    for (blocki = 0; blocki < NETA_LOGBLOCKMAXCOUNT_PERGROUP; blocki++) {
        plogblock = &i_pgroup->parraylogblock[blocki];
        mutex_destroy(&plogblock->producerthreadinfo.mutexthreadinfo);
    }
    if (i_ppagegroup) {
        for (pagei = 0; pagei < (1 << orderblockgroup); pagei++) {
            ClearPageReserved(i_ppagegroup + pagei);
        }
        free_pages((u64)i_pgroup, orderblockgroup);
    }
    return 0;
}

s32 neta_logblockgroup_neednewgroup(u32 i_newgroupindex)
{
    s32 ret = 0;

    if (unlikely(i_newgroupindex >= NETA_LOGBLOCKGROUP_MAXCOUNT)) {
        ret = NETA_KERR_LOGBLOCKGROUP_EXCEEDMAXCOUNT;
        NETA_KLOG_ERR("i_newgroupindex=%u, exceedmaxcount[%u]\n", i_newgroupindex, NETA_LOGBLOCKGROUP_MAXCOUNT);
        return ret;
    }
    if (__neta_log.groupcount >= i_newgroupindex + 1)
    {
        /*
        * Already new the group. Do not need to new it again.
        */
        if (unlikely(__neta_log.groupcount > i_newgroupindex + 1))
        {
            /*
            * Something abnormal.
            */
            NETA_KLOG_INFO("groupcount=%u,newgroupindex=%u\n", __neta_log.groupcount, i_newgroupindex);
        }
        return ret;
    }
    mutex_lock(&__neta_log.mutexnewgroup);
    /*
    * Avoid multi-thread simulateously running new log block group logic cases.
    */
    if (unlikely(__neta_log.groupcount >= i_newgroupindex + 1))
    {
        /*
        * Already new the group. Do not need to new it again.
        */
        if (unlikely(__neta_log.groupcount > i_newgroupindex + 1))
        {
            /*
            * Something abnormal.
            */
            NETA_KLOG_INFO("groupcount=%u,newgroupindex=%u\n", __neta_log.groupcount, i_newgroupindex);
        }
        mutex_unlock(&__neta_log.mutexnewgroup);
        return ret;
    }
    /*
    * Check abnormal cases.
    */
    if (unlikely(__neta_log.groupcount < i_newgroupindex))
    {
        mutex_unlock(&__neta_log.mutexnewgroup);
        ret = NETA_KERR_LOGBLOCKGROUP_NEEDNEW_INDEX_WRONG;
        NETA_KLOG_ERR("ret=%d[%s]: groupcount[%u] < newgroupindex[%u]", ret, neta_kerr_info(ret),
            __neta_log.groupcount, i_newgroupindex);
        return ret;
    }
    /*
    * Run here only when __neta_log.groupcount == i_newgroupindex.
    */
    /*
    * Init the index i_newgroupindex log block group by reserved page mode as default.
    * The second parameter 0 means the group index 0.
    * The third parameter 0 means not reserved page mode.
    */
    CHK_SENTENCE_RET(neta_logblockgroup_create(&__neta_log.parraypgroup[i_newgroupindex],
        &__neta_log.parrayppagegroup[i_newgroupindex],
        &__neta_log.parraypgroupshare[i_newgroupindex],
        &__neta_log.parrayppagegroupshare[i_newgroupindex], i_newgroupindex, 0),
        "neta_logblockgroup_create group");
    /*
    * Add DMB before set groupcount.
    */
    NETA_DMB;
    __neta_log.groupcount = i_newgroupindex + 1;
    mutex_unlock(&__neta_log.mutexnewgroup);

    return ret;
}

s32 neta_logblockgroup_driverinit()
{
    /*
    * When here in the init function, no log block user. So do not need to add
    * any mutex here. Because we create the log node after all of the init
    * function is done.
    */
    s32 ret = 0;

    /*
    * Check whether the shift operation is ok.
    */
    if (unlikely(NETA_LOGBLOCKMAXCOUNT_PERBUDDY != (1 << NETA_LOGBLOCKMAXCOUNT_PERBUDDY_POWER))) {
        ret = NETA_KERR_LOGBLOCKMAXCOUNT_PERBUDDY_POWER_CHECK;
        NETA_KLOG_ERR("ret=%d[%s] power check fail!\n", ret, neta_kerr_info(ret));
        return ret;
    }
    if (unlikely(NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP != (1 << NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP_POWER))) {
        ret = NETA_KERR_LOGBLOCKBUDDYMAXCOUNT_PERGROUP_POWER_CHECK;
        NETA_KLOG_ERR("ret=%d[%s] power check fail!\n", ret, neta_kerr_info(ret));
        return ret;
    }

    /*
    * Init mutex first.
    */
    mutex_init(&__neta_log.mutexnewgroup);

    __neta_log.expectedgroupindex = 0;  // find from the first log group

    __neta_log.lastiteratetimens = neta_gettscns();

    /*
    * Init wait queue.
    */
    init_waitqueue_head(&__neta_log.tagdirtywait);

    /*
    * 0 means need new log block group 0.
    * The group count is changed at last in function neta_logblockgroup_neednewgroup.
    */
    neta_logblockgroup_neednewgroup(0);

    return ret;
}

s32 neta_logblockgroup_driverdeinit()
{
    u32 groupi;
    for (groupi = 0; groupi < __neta_log.groupcount; groupi ++)
    {
        neta_logblockgroup_destroy(__neta_log.parraypgroup[groupi], __neta_log.parrayppagegroup[groupi],
            __neta_log.parraypgroupshare[groupi], __neta_log.parrayppagegroupshare[groupi]);
    }
    /*
    * Destroy mutex in the end.
    */
    mutex_destroy(&__neta_log.mutexnewgroup);
    return 0;
}

s32 neta_logblock_tryget_in_buddy(neta_logblock** o_pplogblock, u32 i_groupindex, u32 i_buddyindex, u32 i_trytimes, u32* o_pbget)
{
    /*
    * Frequently called, performance enhanced. Less check logic.
    */
    s32 ret = 0;
    u32 trytimes = (i_trytimes == 0 ? NETA_LOGBLOCKMAXCOUNT_PERBUDDY : i_trytimes);
    u32 blockindex;
    neta_logblockgroup* pgroup = __neta_log.parraypgroup[i_groupindex];
    neta_logblockbuddy* pbuddy = &__neta_log.parraypgroup[i_groupindex]->parraylogblockbuddy[i_buddyindex];
    neta_logblock* pblock;
    u32 referenceold;
    *o_pplogblock = NULL;
    *o_pbget = 0;
    /*
    * First check wether there is vacant log block.
    */
    if (unlikely(pbuddy->usedcountinbuddy == NETA_LOGBLOCKMAXCOUNT_PERBUDDY)) {
        /*
        * No any vacant log block.
        */
        return 0;
    }
    if (unlikely(i_trytimes > NETA_LOGBLOCKMAXCOUNT_PERBUDDY)) {
        trytimes = NETA_LOGBLOCKMAXCOUNT_PERBUDDY;
    }
    do {
        blockindex = (u32)atomic_add_return(1, (atomic_t*)&pbuddy->expectedblockindexofbuddy) - 1;
        /*
        * First calculate the block index in the buddy, and then calculate the block index of the group.
        */
        blockindex = NETA_LOGBLOCKMAXCOUNT_PERBUDDY * i_buddyindex + (blockindex & (NETA_LOGBLOCKMAXCOUNT_PERBUDDY - 1));
        /*
        * Check wether the specific block index is vacant.
        */
        pblock = &pgroup->parraylogblock[blockindex];
        /*
        * atomic_cmpxchg return the origin value
        * Here, return 0 means get log block success.
        */
        referenceold = (u32)atomic_cmpxchg((atomic_t*)&pblock->reference.refcount, 0, 3);
        if (referenceold == 0) {
            /*
            * Get vacant log block success.
            */
            *o_pplogblock = pblock;
            *o_pbget = 1;
            return 0;
        }
        trytimes --;
        if (trytimes == 0) {
            /*
            * Still can not find vacant log block.
            */
            break;
        }
    } while (1);
    return ret;
}

s32 neta_logblock_tryget_in_group(neta_logblock** o_pplogblock, u32 i_groupindex, u32* o_pbget)
{
    /*
    * Frequently called, performance enhanced. Less check logic.
    */
    s32 ret = 0;
    neta_logblockgroup* pgroup;
    u32 trygetbuddycount;
    u32 buddyindex;
    u32 alreadycreateold;
    *o_pplogblock = NULL;
    *o_pbget = 0;
    pgroup = __neta_log.parraypgroup[i_groupindex];
    /*
    */
    if (unlikely(pgroup->usedcountingroup > INTERNAL_STRATEGY_NEEDTHREADCREATEGROUP_IN_GROUP_USEDCOUNT)) {           
        if (unlikely(alreadycreateold == 0)) {
            alreadycreateold = (u32)atomic_cmpxchg((atomic_t*)&pgroup->already_created, 0, 1);
            if (alreadycreateold == 0) {
                pgroup->already_created = 1;
                thread_add_blockgroup(pgroup->groupindex + 1);
            }
        }
    }
    /*
    * First check whether the group has not any vacant log block.
    */
    if (unlikely(pgroup->usedcountingroup == NETA_LOGBLOCKMAXCOUNT_PERGROUP)) {
        return 0;
    }
    /*
    * Use before check strategy.
    * Try INTERNAL_STRATEGY_TRYGETBUDDYCOUNT_TIMES_BEFORE_CHECK_ENOUGH log block buddy before
    * check whether the log block group.
    * Try INTERNAL_STRATEGY_TRYGETINBUDDY_TIMES_BEFORE_CHECK_ENOUGH log block in the buddy
    * before check whether the log block group.
    */
    trygetbuddycount = INTERNAL_STRATEGY_TRYGETBUDDYCOUNT_TIMES_BEFORE_CHECK_ENOUGH;
    do {
        buddyindex = (u32)atomic_add_return(1, (atomic_t*)&pgroup->expectedbuddyindex) - 1;
        buddyindex = (buddyindex & (NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP - 1));
        ret = neta_logblock_tryget_in_buddy(o_pplogblock, i_groupindex, buddyindex,
            INTERNAL_STRATEGY_TRYGETINBUDDY_TIMES_BEFORE_CHECK_ENOUGH, o_pbget);
        CHK_SENTENCE_RET(ret, "tryget logblock in buddy");
        if (*o_pbget == 1) {
            /*
            * Find vacant log block.
            */
            return 0;
        }
        trygetbuddycount --;
        if (trygetbuddycount == 0) {
            /*
            * Still can not find vacant log block.
            */
            break;
        }
    } while (1);
    /*
    * Check whether has enough vacant log block in the log block group.
    */
    if (pgroup->usedcountingroup > INTERNAL_STRATEGY_NOENOUGHVACANTLOGBLOCK_IN_GROUP_USEDCOUNT) {
        /*
        * No enough vacant log block.
        */
        return 0;
    }
    /*
    * Use after check strategy.
    * Run here means there are enough vacant log blocks.
    */
    trygetbuddycount = INTERNAL_STRATEGY_TRYGETINBUDDY_TIMES_AFTER_CHECK_ENOUGH;
    do {
        buddyindex = (u32)atomic_add_return(1, (atomic_t*)&pgroup->expectedbuddyindex) - 1;
        buddyindex = (buddyindex & (NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP - 1));
        ret = neta_logblock_tryget_in_buddy(o_pplogblock, i_groupindex, buddyindex,
            INTERNAL_STRATEGY_TRYGETBUDDYCOUNT_TIMES_AFTER_CHECK_ENOUGH, o_pbget);
        CHK_SENTENCE_RET(ret, "tryget logblock in buddy");
        if (*o_pbget == 1) {
            /*
            * Find vacant log block.
            */
            return 0;
        }
        trygetbuddycount --;
        if (trygetbuddycount == 0) {
            /*
            * Still can not find vacant log block.
            */
            break;
        }
    } while (1);
    /*
    * When run here means can not get vacant log block.
    */
    return 0;
}

s32 neta_logblock_get(neta_logblock** o_pplogblock)
{
    /*
    * Frequently called, performance enhanced. Less check logic.
    */
    u32 expectedgroupindex = __neta_log.expectedgroupindex;
    u32 expectedgroupindexold = expectedgroupindex;
    u32 groupcount = __neta_log.groupcount;
    u32 bget;
    u32 trytimes = 0;
    do {
        CHK_SENTENCE_RET(neta_logblock_tryget_in_group(o_pplogblock, expectedgroupindex, &bget)
            , "tryget block in group");
        if (bget == 0) {
            if (expectedgroupindex + 1 >= NETA_LOGBLOCKGROUP_MAXCOUNT) {
                __neta_log.expectedgroupindex = 0;
            }
            else {
                __neta_log.expectedgroupindex = expectedgroupindex + 1;
                if (expectedgroupindex + 1 == INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CHECK_GROUP_INDEX) {
                    thread_clear_group();
                }
            }            
            if (unlikely(groupcount == 1)) {
                /*
                * Need new group. When groupcount is 1, the needed new group index is 1.
                */
                CHK_SENTENCE_RET(neta_logblockgroup_neednewgroup(1), "need new group[1]");
                // reset the stack temp variable to new value group index 1
                expectedgroupindex = expectedgroupindexold = 1;
                groupcount = __neta_log.groupcount;
                NETA_DMB;
                __neta_log.expectedgroupindex = 1;
                continue;
            }
            /*
            * When run here, groupcount > 1.
            */
            expectedgroupindex ++;
            if (expectedgroupindex == groupcount) {
                expectedgroupindex = 0;
            }
            /*
            * When already check a total loop, we need new group.
            */
            if (expectedgroupindex == expectedgroupindexold) {
#if (NETA_LOGBLOCK_GET_SLOW_PATH_ENABLE == 1)
                if (unlikely(groupcount >= NETA_LOGBLOCKGROUP_MAXCOUNT)) {
                    NETA_DMB;
                    if (trytimes || __neta_log.bclearinggroup == 0) {
                        clear_group(NULL);
                    }
                    /*
                    * Back to find vacant logblock again.
                    * Currently, set the expectedgroupindex to the beginning.
                    */
                    if (trytimes >= INTERNAL_STRATEGY_ITERATEGETBLOCK_TRYTIMES) {
                        /*
                        * It may trigger exceed max group count error here.
                        */
                        CHK_SENTENCE_RET(neta_logblockgroup_neednewgroup(groupcount), "need new group[>1]");
                        return 0;
                    }
                    trytimes++;
                    __neta_log.expectedgroupindex = 0;
                    expectedgroupindex = __neta_log.expectedgroupindex;
                    expectedgroupindexold = expectedgroupindex;
                    continue;
                }
#else
                /*
                * Need new group. Use stack temp groupcount value.
                */
                CHK_SENTENCE_RET(neta_logblockgroup_neednewgroup(groupcount), "need new group[>1]");
#endif
                // reset the stack temp variable to new value group index groupcount
                expectedgroupindex = expectedgroupindexold = groupcount;
                groupcount = __neta_log.groupcount;
            }
            continue;
        }
        /*
        * When run here, bget == 1. Already find the vacant log block.
        */
        return 0;
    } while (1);
}

s32 neta_logblock_tryput(neta_logblock* i_plogblock, u32* o_pbput)
{
    neta_logblockgroup* pgroup = i_plogblock->groupentry.pgroup;
    if (neta_logblock_returnbdirty(pgroup, i_plogblock->blockindex)) {
        // still tag dirty, put fail
        *o_pbput = 0;
        return 0;
    }
    /*
    * Check whether the read write offset are equal before put the logblock.
    */
    if (i_plogblock->pwroffset->woffset != i_plogblock->pwroffset->roffset) {
        neta_logblock_tagdirty(i_plogblock->groupentry.pgroup, i_plogblock->blockindex);
        *o_pbput = 0;
        return 0;
    }
    /*
    * Check refcount, it should be 2.
    */
    if (unlikely(i_plogblock->reference.refcount != 2)) {
        NETA_KLOG_ERR("netalog logblock tryput expected refcount[%u], should be 2!\n", i_plogblock->reference.refcount);
        return NETA_KERR_LOGBLOCKUSER_TRYPUT_CHECKREFCOUNT_NOT_2;
    }
    /*
    * Currently, we do not change 2 to 1, to mean we now putting the log block back.
    * neta_logblock_put will set to 0 directly after doing all of the put back operation.
    */
    neta_logblock_put(i_plogblock);
    *o_pbput = 1;
    return 0;
}

s32 neta_logblock_put(neta_logblock* i_plogblock)
{
    neta_logblockgroup* pgroup;
    neta_logblockbuddy* pbuddy;
    /*
    * Clear the woffset and roffset.
    */
    i_plogblock->pwroffset->woffset = i_plogblock->pwroffset->roffset = 0;
    // ensure clear the consumer count, may not need to clear it here
    i_plogblock->reference.consumercount = 0;
    pgroup = i_plogblock->groupentry.pgroup;
    pbuddy = i_plogblock->buddyentry.pbuddy;
    atomic_sub_return(1, (atomic_t*)&pgroup->usedcountingroup);
    atomic_sub_return(1, (atomic_t*)&pbuddy->usedcountinbuddy);
    NETA_DMB;
    i_plogblock->reference.refcount = 0;
    return 0;
}

s32 neta_logblock_iteratecheckwprp(void)
{
    u32 groupindex = 0;
    u32 buddyi, blockiperbuddy, blocki;
    neta_logblockgroup* pgroup;
    neta_logblockbuddy* pbuddy;
    neta_logblock* plogblock;
    while (1) {
        if (groupindex >= __neta_log.groupcount) {
            return 0;
        }
        pgroup = __neta_log.parraypgroup[groupindex];
        if (pgroup == NULL) {
            return 0;
        }
        if (pgroup->usedcountingroup > 0) {
            for (buddyi = 0; buddyi < NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP; buddyi++) {
                pbuddy = &pgroup->parraylogblockbuddy[buddyi];
                if (pbuddy->usedcountinbuddy > 0) {
                    for (blockiperbuddy = 0; blockiperbuddy < NETA_LOGBLOCKMAXCOUNT_PERBUDDY; blockiperbuddy++) {
                        blocki = buddyi * NETA_LOGBLOCKMAXCOUNT_PERBUDDY + blockiperbuddy;
                        plogblock = &pgroup->parraylogblock[blocki];
                        if (plogblock->pwroffset->roffset != plogblock->pwroffset->woffset) {
                            neta_logblock_tagdirty(pgroup, blocki);
                        }
                    }
                }
            }
        }
        groupindex++;
    }
    return 0;
}

s32 neta_logblock_alreadyhalf(neta_logblock* io_plogblock)
{
    u32 wpalreadyhalf = __neta_log.wpalreadyhalf;
    u32 rpalreadyhalf = __neta_log.rpalreadyhalf;
    u32 currhalfindex = io_plogblock->alreadyhalfindex;
    /*
    * When write 3/4 of the array, we do not write any more.
    * The current strategy is 3/4 of the NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT.
    * You can change it to other strategy value.
    */
    if (wpalreadyhalf - rpalreadyhalf > ((u32)(NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT * 3) >> 2)) {
        // the array is nearly full, do not add new element
        return 0;
    }
#if (NETA_LOGBLOCK_ALREADYHALF_SIMPLE_LOGIC == 1)
    currhalfindex = atomic_add_return(1, (atomic_t*)&__neta_log.wpalreadyhalf) - 1;
    __neta_log.alreadyhalfplogblockarray[currhalfindex & (NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT - 1)] = io_plogblock;
#else
    /*
    * Check whether has already put the logblock into the already half array.
    */
    if ((wpalreadyhalf != __neta_log.rpalreadyhalf)
        && ((currhalfindex - rpalreadyhalf)
        + (wpalreadyhalf - currhalfindex)
        == (wpalreadyhalf - rpalreadyhalf))) {
        if (__neta_log.alreadyhalfplogblockarray[currhalfindex & (NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT - 1)] == io_plogblock) {
            // already put the log block into the already half array
            return 0;
        }
    }
    /*
    * Need to put the logblock into the already half array.
    */
    /*
    * atomic add the wpalreadyhalf
    */
    currhalfindex = atomic_add_return(1, (atomic_t*)&__neta_log.wpalreadyhalf) - 1;
    /*
    * Check it again, ensure the currently added slot is not overflow.
    */
    if (unlikely(currhalfindex - rpalreadyhalf >= NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT - 1)) {
        // near overflow, skip it
        return 0;
    }
    /*
    * Add the logblock to the slot.
    */
    __neta_log.alreadyhalfplogblockarray[currhalfindex & (NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT - 1)] = io_plogblock;
    NETA_DMB;
    __neta_log.breadyarray[currhalfindex & (NETA_LOGBLOCK_TAGDIRTY_ALREADYHALF_BLOCKMAXCOUNT - 1)] = 1;
#endif
    return 0;
}


static void neta_logblock_dmabufops_release(struct dma_buf *i_pdmabuf)
{
    // currently do nothing
}

static s32 neta_logblock_dmabufops_mmap(struct dma_buf *i_pdmabuf, struct vm_area_struct *i_pvma)
{
    neta_logblock* plogblock = (neta_logblock*)i_pdmabuf->priv;
    u64 phyaddr = plogblock->buddyentry.pbuddy->paddrbuddy + plogblock->buddyentry.offset;
    u64 vmstart = i_pvma->vm_start;
    u32 i = 0;
    s32 ret;
    for (i = 0; i < NETA_LOGBLOCK_SIZE / PAGE_SIZE; i++) {
        ret = remap_pfn_range(i_pvma, vmstart, phyaddr >> PAGE_SHIFT, PAGE_SIZE, i_pvma->vm_page_prot);
        if (unlikely(ret)) {
            NETA_KLOG_ERR("netalog group[%u]logblock[%u] remap_pfn_range err[%d]\n",
                plogblock->groupindex, plogblock->blockindex, ret);
            return ret;
        }
        phyaddr += PAGE_SIZE;
        vmstart += PAGE_SIZE;
    }
    return 0;
}

static const struct dma_buf_ops _logblock_dmabufops = {
    .map_dma_buf = neta_dmabufops_mapdmabuf_notsupport,
    .unmap_dma_buf = neta_dmabufops_unmapdmabuf_notsupport,
    .release = neta_logblock_dmabufops_release,
    .mmap = neta_logblock_dmabufops_mmap,
};

s32 neta_logblock_dmabufgetfd(neta_logblock* io_plogblock, s32* o_pdmabuffd)
{
    DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
    struct dma_buf *pdmabuf;
    exp_info.ops = &_logblock_dmabufops;
    exp_info.size = NETA_LOGBLOCK_SIZE;
    exp_info.flags = O_CLOEXEC | O_RDWR;
    exp_info.priv = io_plogblock;
    pdmabuf = dma_buf_export(&exp_info);
    if (IS_ERR(pdmabuf)) {
        NETA_KLOG_ERR("netalog logblock dma_buf_export fail! dmabuf=[%lld]!\n", (s64)pdmabuf);
        return NETA_KERR_LOGBLOCKUSER_LOGBLOCK_DMABUFEXPORT_FAIL;
    }
    *o_pdmabuffd = dma_buf_fd(pdmabuf, O_CLOEXEC | O_RDWR);
    return 0;
}

static void neta_logblockgroupcontrolinfo_dmabufops_release(struct dma_buf *i_pdmabuf)
{
    // currently do nothing
}

static s32 neta_logblockgroupcontrolinfo_dmabufops_mmap(struct dma_buf *i_pdmabuf, struct vm_area_struct *i_pvma)
{
    u32 groupindex = ((u32)(u64)(i_pdmabuf->priv)) - 1;
    u32 size = PAGE_SIZE * (1 << (get_order(sizeof(neta_logblockgroupshare))));
    u64 phyaddr = (u64)page_to_phys(__neta_log.parrayppagegroupshare[groupindex]);
    u64 vmstart = i_pvma->vm_start;
    u32 i = 0;
    s32 ret;
    for (i = 0; i < size / PAGE_SIZE; i++) {
        ret = remap_pfn_range(i_pvma, vmstart, phyaddr >> PAGE_SHIFT, PAGE_SIZE, i_pvma->vm_page_prot);
        if (unlikely(ret)) {
            NETA_KLOG_ERR("netalog logblockgroup[%u] remap_pfn_range index[%u] err[%d]\n",
                groupindex, i, ret);
            return ret;
        }
        phyaddr += PAGE_SIZE;
        vmstart += PAGE_SIZE;
    }
    return 0;
}

static const struct dma_buf_ops _logblockgroupcontrolinfo_dmabufops = {
    .map_dma_buf = neta_dmabufops_mapdmabuf_notsupport,
    .unmap_dma_buf = neta_dmabufops_unmapdmabuf_notsupport,
    .release = neta_logblockgroupcontrolinfo_dmabufops_release,
    .mmap = neta_logblockgroupcontrolinfo_dmabufops_mmap,
};

s32 neta_logblockgroupcontrolinfo_dmabufgetfd(neta_lognodeprocessproducer* i_pprocessproducer, u32 i_groupindex, s32* o_pdmabuffd)
{
    DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
    struct dma_buf *pdmabuf;
    s32 groupsharefd;
    mutex_lock(&i_pprocessproducer->mutexgroupsharefd);
    if (i_pprocessproducer->parraygroupsharefd[i_groupindex] != 0) {
        *o_pdmabuffd = i_pprocessproducer->parraygroupsharefd[i_groupindex];
        mutex_unlock(&i_pprocessproducer->mutexgroupsharefd);
        return 0;
    }
    exp_info.ops = &_logblockgroupcontrolinfo_dmabufops;
    exp_info.size = PAGE_SIZE * (1 << (get_order(sizeof(neta_logblockgroupshare))));
    exp_info.flags = O_CLOEXEC | O_RDWR;
    // add 1 for not trigger WARN_ON check in dma_buf_export
    exp_info.priv = (void*)(u64)(i_groupindex + 1);
    pdmabuf = dma_buf_export(&exp_info);
    if (IS_ERR(pdmabuf)) {
        NETA_KLOG_ERR("netalog logblockgroupcontrolinfo dma_buf_export fail! dmabuf=[%lld]!\n", (s64)pdmabuf);
        mutex_unlock(&i_pprocessproducer->mutexgroupsharefd);
        return NETA_KERR_LOGBLOCKUSER_LOGBLOCKGROUPSHARE_DMABUFEXPORT_FAIL;
    }
    groupsharefd = dma_buf_fd(pdmabuf, O_CLOEXEC | O_RDWR);
    *o_pdmabuffd = groupsharefd;
    i_pprocessproducer->parraygroupsharefd[i_groupindex] = groupsharefd;
    mutex_unlock(&i_pprocessproducer->mutexgroupsharefd);
    return 0;
}

static void neta_logblockgroup_dmabufops_release(struct dma_buf *i_pdmabuf)
{
    // currently do nothing
}

static s32 neta_logblockgroup_dmabufops_mmap(struct dma_buf *i_pdmabuf, struct vm_area_struct *i_pvma)
{
    u32 groupindex = ((u32)(u64)(i_pdmabuf->priv)) - 1;
    u64 phyaddr;
    u64 vmstart = i_pvma->vm_start;
    u32 buddyi;
    u32 i;
    s32 ret;
    for (buddyi = 0; buddyi < NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP; buddyi++) {
        phyaddr = __neta_log.parraypgroup[groupindex]->parraylogblockbuddy[buddyi].paddrbuddy;
        for (i = 0; i < NETA_BUDDY_MAXSIZE / PAGE_SIZE; i++) {
            ret = remap_pfn_range(i_pvma, vmstart, phyaddr >> PAGE_SHIFT, PAGE_SIZE, i_pvma->vm_page_prot);
            if (unlikely(ret)) {
                NETA_KLOG_ERR("netalog logblockgroup[%u] remap_pfn_range buddyi[%u]index[%u] err[%d]\n",
                    groupindex, buddyi, i, ret);
                return ret;
            }
            phyaddr += PAGE_SIZE;
            vmstart += PAGE_SIZE;
        }
    }
    return 0;
}

static const struct dma_buf_ops _logblockgroup_dmabufops = {
    .map_dma_buf = neta_dmabufops_mapdmabuf_notsupport,
    .unmap_dma_buf = neta_dmabufops_unmapdmabuf_notsupport,
    .release = neta_logblockgroup_dmabufops_release,
    .mmap = neta_logblockgroup_dmabufops_mmap,
};

s32 neta_logblockgroup_dmabufgetfd(neta_lognodeprocessconsumer* i_pprocessconsumer, u32 i_groupindex, s32* o_pdmabuffd)
{
    DEFINE_DMA_BUF_EXPORT_INFO(exp_info);
    struct dma_buf *pdmabuf;
    s32 groupfd;
    if (likely(i_pprocessconsumer->parraygroupfd[i_groupindex] != 0)) {
        *o_pdmabuffd = i_pprocessconsumer->parraygroupfd[i_groupindex];
        return 0;
    }
    mutex_lock(&i_pprocessconsumer->mutexgroupfd);
    if (unlikely(i_pprocessconsumer->parraygroupfd[i_groupindex] != 0)) {
        *o_pdmabuffd = i_pprocessconsumer->parraygroupfd[i_groupindex];
        mutex_unlock(&i_pprocessconsumer->mutexgroupfd);
        return 0;
    }
    exp_info.ops = &_logblockgroup_dmabufops;
    exp_info.size = NETA_LOGBLOCKGROUP_SIZE;
    exp_info.flags = O_CLOEXEC | O_RDWR;
    exp_info.priv = (void*)(u64)(i_groupindex + 1);
    pdmabuf = dma_buf_export(&exp_info);
    if (unlikely(IS_ERR(pdmabuf))) {
        NETA_KLOG_ERR("netalog logblockgroup dma_buf_export fail! dmabuf=[%lld]!\n", (s64)pdmabuf);
        mutex_unlock(&i_pprocessconsumer->mutexgroupfd);
        return NETA_KERR_LOGBLOCKUSER_LOGBLOCKGROUPSHARE_DMABUFEXPORT_FAIL;
    }
    groupfd = dma_buf_fd(pdmabuf, O_CLOEXEC | O_RDWR);
    *o_pdmabuffd = groupfd;
    NETA_DMB;
    i_pprocessconsumer->parraygroupfd[i_groupindex] = groupfd;
    mutex_unlock(&i_pprocessconsumer->mutexgroupfd);
    return 0;
}
