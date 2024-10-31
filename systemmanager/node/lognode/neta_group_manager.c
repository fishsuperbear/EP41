#include <linux/sched.h>
#include <linux/kthread.h>

#include "neta_log.h"

static struct task_struct *pthread[5];
static struct task_struct *clrthread;

static DEFINE_MUTEX(_mutexcleargroup); 

int alloc_new_group(void *data)
{
    u32 groupindex;
    s32 ret;
    groupindex = __neta_log.groupcount;
    if (groupindex >= NETA_LOGBLOCKGROUP_MAXCOUNT) {
        return 0;
    }
    ret = neta_logblockgroup_neednewgroup(groupindex);
    if(ret !=0){
        printk(KERN_INFO"alloc new group%d failed \n",groupindex);
        return 0;
    }
    return 0;
}

void thread_add_blockgroup(u32 i_groupindex)
{
    if(i_groupindex == NETA_LOGBLOCKGROUP_MAXCOUNT) {
        return;
    }
    pthread[i_groupindex] = kthread_run(alloc_new_group, NULL, "knewgroup%d", i_groupindex);
    if(pthread[i_groupindex] == NULL){
        NETA_KLOG_ERR("create kthread err \n");
        return;
    }
    return;
} 

s32 clear_group(void* i_param)
{
    neta_logblock* pblock;
    u32 consumercountold;
    u32 blockindex;
    neta_logblockgroup* pgroup;
    __neta_log.bclearinggroup = 1;
    mutex_lock(&_mutexcleargroup);
    pgroup = __neta_log.parraypgroup[__neta_log.grouptoclearindex];
    if (__neta_log.grouptoclearindex == INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CLEAR_GROUP_INDEX_END) {
        __neta_log.grouptoclearindex = INTERNAL_STRATEGY_NEEDTHREADPUTBLOCK_CLEAR_GROUP_INDEX_BEGIN;
    }
    else {
        __neta_log.grouptoclearindex++;
    }
    if (pgroup->usedcountingroup > NETA_LOGBLOCKMAXCOUNT_PERGROUP / 16) {
        for(blockindex = 0; blockindex < NETA_LOGBLOCKMAXCOUNT_PERGROUP; blockindex++){
            pblock = &pgroup->parraylogblock[blockindex];
            if (pblock->reference.refcount == 2) {
                consumercountold = (u32)atomic_cmpxchg((atomic_t*)&pblock->reference.consumercount, 0, 1);
                if (consumercountold != 0) {
                    continue;
                }
                neta_logblock_tagnotdirty(pgroup, blockindex);
                neta_logblock_put(pblock);
            }
        }
    }
    mutex_unlock(&_mutexcleargroup);
    __neta_log.bclearinggroup = 0;
    return 0;
}
void thread_clear_group(void)
{
    if (__neta_log.bclearinggroup == 0) {
        __neta_log.bclearinggroup = 1;
        NETA_DMB;
        clrthread = kthread_run(clear_group, NULL, "kcleargroup");
        if(clrthread == NULL){
            NETA_KLOG_ERR("create kthread err \n");
            return;
        }
    }
}