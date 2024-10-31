#ifndef NETA_LOGNODE_H
#define NETA_LOGNODE_H

#include "neta_lognode_ioctl.h"

#define NETA_LOGBASENODE_NAME              "netalog"
#define NETA_LOGGLOBALINFONODE_NAME        "global_info"

typedef struct neta_logblock neta_logblock;

typedef struct neta_logblockdevnode
{
    struct class*       char_class;
	struct cdev         cdev;
	dev_t               devno;
    s32                 majordevnum;
} neta_logblockdevnode;

typedef struct neta_lognode
{
    struct proc_dir_entry*      pprocentry_logbase;
    // output dynamic info about log
    struct proc_dir_entry*      pprocentry_logglobalinfo;
    /*
    * The dev node for log block.
    * The user role of the dev node is log block producer only or log block consumer 
    * only. It is not permitted that single fd support not only producer but also 
    * consumer. You need to open two fd seperately correspondent to producer only and 
    * consumer only.
    */
    neta_logblockdevnode*       plogblockdevnode;
} neta_lognode;

/*
* Log node user part, producer only or consumer only.
*/

/*
* The user of a specific process producer thread. Managed by neta_lognodeprocessproducer.
*/
typedef struct neta_lognodethreadproducer
{
    // process id, get by getpid() in user mode
    s32                 processid;
    // thread id, get by gettid() in user mode
    s32                 threadid;
    // the time(ns unit) when do get log block ioctl, get by hw_plat_get_tsc_ns in user mode.
    u64                 tscns;
    char                name[LOGBLOCK_NAME_LENGTH];
    u32                 groupindex;
    u32                 blockindex;
    neta_logblock*      plogblock;
    // add it to threadproducerlist of neta_lognodeprocessproducer
    struct list_head    threadproducernode;
} neta_lognodethreadproducer;

typedef struct neta_lognodeprocessproducer
{
    // init when malloc neta_lognodeprocessproducer
    struct mutex        mutexthreadproducerlist;
    // init when malloc neta_lognodeprocessproducer
    struct list_head    threadproducerlist;
    // init when malloc neta_lognodeprocessproducer
    struct mutex        mutexgroupsharefd;
    // one process only has one groupsharefd every group. 0 as default value.
    s32                 parraygroupsharefd[NETA_LOGBLOCKGROUP_MAXCOUNT];
} neta_lognodeprocessproducer;

#if (NETA_KLOG_ENABLE_COMBINENS_DEBUG == 1)
#define INTERNAL_NETA_THREADCONSUMER_COMBINE_MAX    4
#endif

/*
* The user of a specific process consumer thread. Managed by neta_lognodeprocessconsumer.
*/
typedef struct neta_lognodethreadconsumer
{
    // process id, get by current->tgid in kernel mode
    s32                 processid;
    // thread id, get by current->pid in kernel mode
    s32                 threadid;
    // decide the schedule strategy
    u32                 readerthreadid;
    // NULL means already sub reference, not NULL means is consuming and need to sub reference later
    neta_logblock*      plogblock;
    // for update the consumer reader point use only, valid when plogblock is not NULL
    u32                 roffset_toread;
    // the next dirty combine u64 group index to check, 0~groupcount-1
    u32                 u64dirtybitgroupindex;
    // the next dirty combine u64 index to check of specific group, 0~NETA_LOGBLOCK_DIRTYBIT_U64ARRAYMAXCOUNT-1
    u32                 u64dirtybitcombineindex;
    // the bit index of the u64 element, 0~63
    u32                 bitindex;
    // the alternately get the logblock from alreadyhalf array and get the logblock by round robin
    u32                 bcurrroundrobin;
#if (NETA_ENABLE_HIGH_PERFORMANCE_MODE == 1)
    // the origin sched policy
    s32                 policyori;
    // the origin nice when policy is SCHED_NORMAL, valid when policyori is SCHED_NORMAL
    s32                 niceori;
    // the current sched policy
    s32                 policycurr;
    // every getnextlogblocktoread get the value from 
    u32                 alreadyhalflastcount;
    // the last time ns when need to set sched fifo
    u64                 lastnsneedfifo;
#endif
#if (NETA_KLOG_ENABLE_COMBINENS_DEBUG == 1)
    u64                 debugnsbeforecombinearray[INTERNAL_NETA_THREADCONSUMER_COMBINE_MAX];
    u64                 debugnsaftercombinearray[INTERNAL_NETA_THREADCONSUMER_COMBINE_MAX];
    u64                 checkwprpcount;
#endif
    // add it to threadconsumerlist of neta_lognodeprocessconsumer
    struct list_head    threadconsumernode;
} neta_lognodethreadconsumer;

typedef struct neta_lognodeprocessconsumer
{
    // init when malloc neta_lognodeprocessconsumer
    struct mutex        mutexthreadconsumerlist;
    // init when malloc neta_lognodeprocessconsumer
    struct list_head    threadconsumerlist;
    // init when malloc neta_lognodeprocessconsumer
    struct mutex        mutexgroupfd;
    // one process only has one groupfd every group. 0 as default value.
    s32                 parraygroupfd[NETA_LOGBLOCKGROUP_MAXCOUNT];
} neta_lognodeprocessconsumer;

/*
* The user of a specific process.
* One process can has at most two users, one as neta_lognodethreadproducer, 
* the other as consumer.
* But the neta_lognodeprocessuser can only has one of the two type instance.
* Store the instance in pfile->private_data.
* Currently, one process has at most only one dev/netahal producer node and has at most 
* only one dev/netahal consumer node.
* All of the log node thread user in the process user should be the same type.
*/
typedef struct neta_lognodeprocessuser
{
    /*
    * Init when neta_lognodeprocessuser init.
    * Protect the bproducer and the constructor of neta_lognodeprocessproducer or
    * neta_lognodeprocessconsumer. Because once open the node, there are may be 
    * more than one thread ioctl at the same time.
    */
    struct mutex                    mutexprocess;
    /*
    * -1, 0 or 1, 1 means log producer, 0 means log consumer
    * -1 means it is not certain whether the role is consumero or producer.
    * pprocessproducer and pprocessconsumer cannot be valid both
    */
    u32                             bproducer;
    /*
    * Default is 0, when pre-map one fd, then plus one.
    */
    u32                             premapgroupcount;
    neta_lognodeprocessproducer*    pprocessproducer;
    neta_lognodeprocessconsumer*    pprocessconsumer;
} neta_lognodeprocessuser;

int neta_proc_logbasenode_create(void);
int neta_proc_logbasenode_destroy(void);
int neta_proc_logglobalinfonode_create(void);
int neta_proc_logglobalinfonode_destroy(void);

int neta_proc_logglobalinfonode_open(struct inode *i_pinode, struct file *i_pfile);
int neta_proc_logglobalinfonode_show(struct seq_file *i_pseqfile, void *i_private);

/*
* The following function is about device node of log block.
*/

/*
* Init the dev log node for log block use.
*/
s32 neta_dev_logblocknode_create(void);

/*
* Destroy the dev log node for log block use.
*/
s32 neta_dev_logblocknode_destroy(void);

#endif
