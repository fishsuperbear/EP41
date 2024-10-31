#include "neta_log.h"

s32 internal_neta_dev_logblocknode_ioctl_getlogblock(neta_lognodeprocessuser* io_pprocessuser,
    neta_logwriterioctl_getlogblock* io_pgetlogblock)
{
    neta_lognodeprocessproducer* pproducer;
    s32 ret = 0;
    neta_logblock* plogblock;
    s32 dmabuffdlogblock;
    s32 dmabuffdgroupshare;
    u32 groupi;
    neta_lognodethreadproducer* pthreadproducer;
    /*
    * First, do some basic check.
    */
    if (unlikely(io_pgetlogblock->io_writerthread_cookie != 0)) {
        NETA_KLOG_ERR("netalog ioctl getlogblock io_writerthread_cookie[0x%llx] not 0!\n", io_pgetlogblock->io_writerthread_cookie);
        return NETA_KERR_LOGBLOCKUSER_THREADCOOKIE_ALREADY_NOT_0;
    }
    if (likely(io_pprocessuser->bproducer == 1)) {
        pproducer = io_pprocessuser->pprocessproducer;
    }
    else {
        mutex_lock(&io_pprocessuser->mutexprocess);
        if (unlikely(io_pprocessuser->bproducer == 0)) {
            NETA_KLOG_ERR("netalog ioctl getlogblock when bproducer[%u]!\n", io_pprocessuser->bproducer);
            mutex_unlock(&io_pprocessuser->mutexprocess);
            ret = NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLCONSUMERFUNC_WHENPRODUCER;
            io_pgetlogblock->o_ret = ret;
            return ret;
        }
        /*
        * The current thread role is log block producer. When it is not, we need to
        * do the init operation.
        */
        if (io_pprocessuser->bproducer == -1) {
            /*
            * Init the process producer env.
            */
            pproducer = kzalloc(sizeof(neta_lognodeprocessproducer), GFP_KERNEL);
            mutex_init(&pproducer->mutexthreadproducerlist);
            INIT_LIST_HEAD(&pproducer->threadproducerlist);
            mutex_init(&pproducer->mutexgroupsharefd);
            for (groupi = 0; groupi < NETA_LOGBLOCKGROUP_MAXCOUNT; groupi++) {
                pproducer->parraygroupsharefd[groupi] = 0;
            }
            io_pprocessuser->pprocessproducer = pproducer;
            NETA_DMB;
            io_pprocessuser->bproducer = 1;
            /*
            * Add the log writer process count.
            */
            atomic_add_return(1, (atomic_t*)&__neta_log.writerprocesscount);
        }
        else {
            /*
            * Need the following operation to ensure the operation is done.
            */
            pproducer = io_pprocessuser->pprocessproducer;
        }
        mutex_unlock(&io_pprocessuser->mutexprocess);
    }
    /*
    * Get log block.
    */
    ret = neta_logblock_get(&plogblock);
    if (unlikely(ret < 0)) {
        io_pgetlogblock->o_ret = ret;
        return ret;
    } else {
        atomic_add_return(1, (atomic_t*)&plogblock->buddyentry.pbuddy->usedcountinbuddy);
        atomic_add_return(1, (atomic_t*)&plogblock->groupentry.pgroup->usedcountingroup);
    }
    /*
    * Successfully get a vacant log block.
    */
#if (NETA_KLOG_ENABLE_DEBUG_LOG == 1)
    NETA_KLOG_INFO("logblock groupindex[%u]blockindex[%u]\n", plogblock->groupindex, plogblock->blockindex);
#endif
    /*
    * Set the producer thread info to the logblock.
    */
    mutex_lock(&plogblock->producerthreadinfo.mutexthreadinfo);
    plogblock->producerthreadinfo.bproducing = 1;
    plogblock->producerthreadinfo.processid = io_pgetlogblock->i_processid;
    plogblock->producerthreadinfo.threadid = io_pgetlogblock->i_threadid;
    plogblock->producerthreadinfo.tscns = io_pgetlogblock->i_tscns;
    memcpy(plogblock->producerthreadinfo.name, io_pgetlogblock->i_name, LOGBLOCK_NAME_LENGTH);
    mutex_unlock(&plogblock->producerthreadinfo.mutexthreadinfo);
    /*
    * We add neta_lognodethreadproducer to the list managed by neta_lognodeprocessuser.
    */
    pthreadproducer = kzalloc(sizeof(neta_lognodethreadproducer), GFP_KERNEL);
    pthreadproducer->processid = io_pgetlogblock->i_processid;
    pthreadproducer->threadid = io_pgetlogblock->i_threadid;
    pthreadproducer->tscns = io_pgetlogblock->i_tscns;
    memcpy(pthreadproducer->name, io_pgetlogblock->i_name, LOGBLOCK_NAME_LENGTH);
    pthreadproducer->groupindex = plogblock->groupindex;
    pthreadproducer->blockindex = plogblock->blockindex;
    pthreadproducer->plogblock = plogblock;
    mutex_lock(&pproducer->mutexthreadproducerlist);
    list_add_tail(&pthreadproducer->threadproducernode, &pproducer->threadproducerlist);
    mutex_unlock(&pproducer->mutexthreadproducerlist);
    io_pgetlogblock->io_writerthread_cookie = (u64)pthreadproducer;
    io_pgetlogblock->o_groupindex = plogblock->groupindex;
    io_pgetlogblock->o_blockindex = plogblock->blockindex;
    ret = neta_logblock_dmabufgetfd(plogblock, &dmabuffdlogblock);
    if (unlikely(ret < 0)) {
        io_pgetlogblock->o_ret = ret;
        return ret;
    }
    io_pgetlogblock->o_logblockfd = dmabuffdlogblock;
    ret = neta_logblockgroupcontrolinfo_dmabufgetfd(pproducer, plogblock->groupindex, &dmabuffdgroupshare);
    if (unlikely(ret < 0)) {
        io_pgetlogblock->o_ret = ret;
        return ret;
    }
    io_pgetlogblock->o_controlinfofd = dmabuffdgroupshare;
    io_pgetlogblock->o_controlinfosize = PAGE_SIZE * (1 << (get_order(sizeof(neta_logblockgroupshare))));
    io_pgetlogblock->o_ret = 0;
    return 0;
}

s32 internal_neta_dev_logblocknode_ioctl_putlogblock(neta_lognodeprocessuser* io_pprocessuser,
    neta_logwriterioctl_putlogblock* io_pputlogblock)
{
    s32 ret = 0;
    neta_lognodeprocessproducer* pproducer;
    u32 bput;
    neta_lognodethreadproducer* pthreadproducer;
    neta_logblock* plogblock;
    u32 referencenew;
    /*
    * The current thread role is log block producer.
    */
    if (unlikely(io_pprocessuser->bproducer != 1)) {
        NETA_KLOG_ERR("netalog ioctl putlogblock when bproducer[%u]!\n", io_pprocessuser->bproducer);
        ret = NETA_KERR_LOGBLOCKUSER_ROLECHECKFAIL_CALLCONSUMERFUNC_WHENPRODUCER;
        io_pputlogblock->o_ret = ret;
        return ret;
    }
    if (unlikely(io_pputlogblock->i_writerthread_cookie == 0)) {
        NETA_KLOG_ERR("netalog ioctl putlogblock io_writerthread_cookie is 0!\n");
        ret = NETA_KERR_LOGBLOCKUSER_THREADCOOKIE_IS_0;
        io_pputlogblock->o_ret = ret;
        return ret;
    }
    pproducer = io_pprocessuser->pprocessproducer;
    pthreadproducer = (neta_lognodethreadproducer*)io_pputlogblock->i_writerthread_cookie;
    plogblock = pthreadproducer->plogblock;
    /*
    * Check plogblock for sure.
    */
    if (plogblock) {
        /*
        * Set the producer thread info to not producing status.
        */
        mutex_lock(&plogblock->producerthreadinfo.mutexthreadinfo);
        plogblock->producerthreadinfo.bproducing = 0;
        mutex_unlock(&plogblock->producerthreadinfo.mutexthreadinfo);
        /*
        * atomic_sub_return return the value after sub operation.
        */
        referencenew = (u32)atomic_sub_return(1, (atomic_t*)&plogblock->reference.refcount);
        if (referencenew == 2) {
            ret = neta_logblock_tryput(plogblock, &bput);
            if (unlikely(ret < 0)) {
                io_pputlogblock->o_ret = ret;
                return ret;
            }
            // do nothing here when bput is 0 or 1
        }
    }
    /*
    * Delete the thread user from the thread user list.
    */
    mutex_lock(&pproducer->mutexthreadproducerlist);
    list_del(&pthreadproducer->threadproducernode);
    mutex_unlock(&pproducer->mutexthreadproducerlist);
    kfree(pthreadproducer);
    io_pputlogblock->o_ret = 0;
    return 0;
}

s32 internal_neta_dev_logblocknode_ioctl_tagdirty(neta_lognodeprocessuser* io_pprocessuser,
    neta_logwriterioctl_tagdirty* io_ptagdirty)
{
    /*
    * No any check here.
    */
    __neta_log.btagdirty = 1;
#if (NETA_KLOG_ENABLE_EXTRA_DEBUG_LOG == 1)
    NETA_KLOG_INFO("time[%llu]tagdirty by ioctl\n", neta_gettscns());
#endif
    /*
    * Check wether need to triger already half logic.
    */
    if (io_ptagdirty->i_groupindex != -1) {
        neta_logblock_alreadyhalf(&__neta_log.parraypgroup[io_ptagdirty->i_groupindex]->parraylogblock[io_ptagdirty->i_blockindex]);
        __neta_log.alreadyhalfcounter++;
    }
    NETA_DMB;
    wake_up_interruptible_all(&__neta_log.tagdirtywait);
    io_ptagdirty->o_ret = 0;
    return 0;
}
