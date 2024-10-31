#include "neta_node.h"
#include "neta_log.h"
#include "neta_lognode_ioctl_common.h"

int neta_proc_logbasenode_create(void)
{
    __neta_node.lognode.pprocentry_logbase = proc_mkdir(NETA_LOGBASENODE_NAME, NULL);
    if (__neta_node.lognode.pprocentry_logbase == NULL) {
        NETA_KLOG_ERR("proc create %s failed!\n", NETA_LOGBASENODE_NAME);
        return -ENOMEM;
    }
    return 0;
}

int neta_proc_logbasenode_destroy(void)
{
    if (__neta_node.lognode.pprocentry_logbase == NULL) {
        NETA_KLOG_ERR("proc %s not exist!\n", NETA_LOGBASENODE_NAME);
        return -ENOENT;
    }
    remove_proc_entry(NETA_LOGBASENODE_NAME, NULL);
    return 0;
}

static const struct proc_ops _fops_proclogglobainfonode = {
    .proc_open    = neta_proc_logglobalinfonode_open,
    .proc_read    = neta_procops_read_default,
    .proc_write   = NULL,
    .proc_lseek   = neta_procops_lseek_default,
    .proc_release = neta_procops_release_default,
};

int neta_proc_logglobalinfonode_create(void)
{
    __neta_node.lognode.pprocentry_logglobalinfo = proc_create(NETA_LOGGLOBALINFONODE_NAME,
        0444, __neta_node.lognode.pprocentry_logbase, &_fops_proclogglobainfonode);
    if (__neta_node.lognode.pprocentry_logglobalinfo == NULL) {
        NETA_KLOG_ERR("proc create %s failed!\n", NETA_LOGGLOBALINFONODE_NAME);
        return -ENOMEM;
    }
    return 0;
}

int neta_proc_logglobalinfonode_destroy(void)
{
    if (__neta_node.lognode.pprocentry_logglobalinfo == NULL) {
        NETA_KLOG_ERR("proc %s not exist!\n", NETA_LOGGLOBALINFONODE_NAME);
        return -ENOENT;
    }
    remove_proc_entry(NETA_LOGGLOBALINFONODE_NAME, __neta_node.lognode.pprocentry_logbase);
    return 0;
}

int neta_proc_logglobalinfonode_open(struct inode *i_pinode, struct file *i_pfile)
{
    return single_open(i_pfile, neta_proc_logglobalinfonode_show, NULL);
}

int neta_proc_logglobalinfonode_show(struct seq_file *i_pseqfile, void *i_private)
{
    u32 groupi, buddyi, blocki;
    u32 groupcount = __neta_log.groupcount;
    neta_logblockgroup* pgroup;
    neta_logblockbuddy* pbuddy;
    neta_logblock* plogblock;
    seq_printf(i_pseqfile, "global_info output begin...\n");
    seq_printf(i_pseqfile, "[build time]: %s %s. para_logmode=%s\n", __DATE__, __TIME__, para_logmode);
    seq_printf(i_pseqfile, "\n");
    seq_printf(i_pseqfile, "**********Total Info**********\n");
    seq_printf(i_pseqfile, "groupcount[%u] writerprocesscount[%u] readerthreadcount[%u]\n", \
        groupcount, __neta_log.writerprocesscount, __neta_log.readerthreadcount);
    seq_printf(i_pseqfile, "expectedgroupindex[%u] expectedreaderthreadid[%u]\n", \
        __neta_log.expectedgroupindex, __neta_log.expectedreaderthreadid);
    if (groupcount > 0) {
        seq_printf(i_pseqfile, "\n");
        seq_printf(i_pseqfile, "**********Group Info**********\n");
        for (groupi = 0; groupi < groupcount; groupi++) {
            pgroup = __neta_log.parraypgroup[groupi];
            if (pgroup != NULL) {
                seq_printf(i_pseqfile, "\n groupindex[%u] usedcountingroup[%u] expectedbuddyindex[%u]\n", \
                    pgroup->groupindex, pgroup->usedcountingroup, pgroup->expectedbuddyindex);
                seq_printf(i_pseqfile, "\n");
                seq_printf(i_pseqfile, "**********Buddy Info**********\n");
                /*
                * Output info of buddys.
                */
                for (buddyi = 0; buddyi < NETA_LOGBLOCKBUDDYMAXCOUNT_PERGROUP; buddyi++) {
                    pbuddy = &pgroup->parraylogblockbuddy[buddyi];
                    seq_printf(i_pseqfile, "buddyindex[%u] usedcountinbuddy[%u] expectedblockindexofbuddy[%u]\n", \
                        buddyi, pbuddy->usedcountinbuddy, pbuddy->expectedblockindexofbuddy);
                }
                if (pgroup->usedcountingroup > 0) {
                    seq_printf(i_pseqfile, "\n");
                    seq_printf(i_pseqfile, "**********Block Info**********\n");
                    /*
                    * Output info of blocks.
                    */
                    for (blocki = 0; blocki < NETA_LOGBLOCKMAXCOUNT_PERGROUP; blocki++) {
                        plogblock = &pgroup->parraylogblock[blocki];
                        if (plogblock->reference.refcount > 0) {
                            seq_printf(i_pseqfile, "blockindex[%u] refcount[%u] consumercount[%u] bdirty[%u] rp[%u] wp[%u]\n", \
                                plogblock->blockindex, \
                                plogblock->reference.refcount, \
                                plogblock->reference.consumercount, \
                                neta_logblock_returnbdirty(pgroup, blocki), \
                                plogblock->pwroffset->roffset, \
                                plogblock->pwroffset->woffset);
                            seq_printf(i_pseqfile, "Producer : bproducing[%u] processid[%d] threadid[%d] tscns[%llu] name[%s]\n", \
                                plogblock->producerthreadinfo.bproducing, \
                                plogblock->producerthreadinfo.processid, \
                                plogblock->producerthreadinfo.threadid, \
                                plogblock->producerthreadinfo.tscns,
                                plogblock->producerthreadinfo.name);
                        }
                    }
                }
            }
        }
    }
    seq_printf(i_pseqfile, "\n");
    seq_printf(i_pseqfile, "global_info output end...\n");
    return 0;
}

/*
* The following function is about device node of log block.
*/

static long internal_neta_dev_logblocknode_ioctl(struct file *i_pfile, u32 i_cmd, long unsigned int i_arg)
{
    neta_lognodeprocessuser* pprocessuser = (neta_lognodeprocessuser*)i_pfile->private_data;
    void __user* parg = (void __user*)i_arg;
    s32 internalret = 0;

    switch(i_cmd){
    case NETA_LOGNODE_IO_LOGWRITER_GETLOGBLOCK:
        {
            neta_logwriterioctl_getlogblock getlogblock;
            if (copy_from_user(&getlogblock, parg, sizeof(getlogblock))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_getlogblock(pprocessuser, &getlogblock);
            if (copy_to_user(parg, &getlogblock, sizeof(getlogblock))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGWRITER_PUTLOGBLOCK:
        {
            neta_logwriterioctl_putlogblock putlogblock;
            if (copy_from_user(&putlogblock, parg, sizeof(putlogblock))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_putlogblock(pprocessuser, &putlogblock);
            if (copy_to_user(parg, &putlogblock, sizeof(putlogblock))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGWRITER_TAGDIRTY:
        {
            neta_logwriterioctl_tagdirty tagdirty;
            if (copy_from_user(&tagdirty, parg, sizeof(tagdirty))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_tagdirty(pprocessuser, &tagdirty);
            if (copy_to_user(parg, &tagdirty, sizeof(tagdirty))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGREADER_GETNEXTLOGBLOCKTOREAD:
        {
            neta_logreaderioctl_getnextlogblocktoread getnextblocktoread;
            if (copy_from_user(&getnextblocktoread, parg, sizeof(getnextblocktoread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_getnextlogblocktoread(pprocessuser, &getnextblocktoread);
            if (copy_to_user(parg, &getnextblocktoread, sizeof(getnextblocktoread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGREADER_FINISHLOGBLOCKREAD:
        {
            neta_logreaderioctl_finishlogblockread fnishlogblockread;
            if (copy_from_user(&fnishlogblockread, parg, sizeof(fnishlogblockread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_finishlogblockread(pprocessuser, &fnishlogblockread);
            if (copy_to_user(parg, &fnishlogblockread, sizeof(fnishlogblockread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGREADER_READERTHREADQUIT:
        {
            neta_logreaderioctl_readerthreadquit readerthreadquit;
            if (copy_from_user(&readerthreadquit, parg, sizeof(readerthreadquit))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_readerthreadquit(pprocessuser, &readerthreadquit);
            if (copy_to_user(parg, &readerthreadquit, sizeof(readerthreadquit))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGREADER_WAKEUPREADERTHREAD:
        {
            neta_logreaderioctl_wakeupreaderthread wakeupreaderthread;
            if (copy_from_user(&wakeupreaderthread, parg, sizeof(wakeupreaderthread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_wakeupreaderthread(pprocessuser, &wakeupreaderthread);
            if (copy_to_user(parg, &wakeupreaderthread, sizeof(wakeupreaderthread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGREADER_CHECKLOGBLOCKPRODUCERTHREAD:
        {
            neta_logreaderioctl_checklogblockproducerthread checklogblockproducerthread;
            if (copy_from_user(&checklogblockproducerthread, parg, sizeof(checklogblockproducerthread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_checklogblockproducerthread(pprocessuser, &checklogblockproducerthread);
            if (copy_to_user(parg, &checklogblockproducerthread, sizeof(checklogblockproducerthread))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    case NETA_LOGNODE_IO_LOGREADER_GETGLOBALSTATUS:
        {
            neta_logreaderioctl_getglobalstatus getglobalstatus;
            if (copy_from_user(&getglobalstatus, parg, sizeof(getglobalstatus))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            internalret = internal_neta_dev_logblocknode_ioctl_getglobalstatus(pprocessuser, &getglobalstatus);
            if (copy_to_user(parg, &getglobalstatus, sizeof(getglobalstatus))) {
                NETA_KLOG_UNEXPECTED_RUN_HERE;
                return -EFAULT;
            }
            break;
        }
    default:
        // when the input cmd is error, just return invalid parameter
        NETA_KLOG_ERR("netalog ioctl i_cmd[%u] not support!\n", i_cmd);
        return -EINVAL;
    }
    if (unlikely(internalret != 0)) {
        NETA_KLOG_ERR("netalog ioctl internalret=%d[%s]\n", internalret, neta_kerr_info(internalret));
    }
    /*
    * Whatever the internal ioctl return is 0 or not, when the cmd is write, the
    * ioctl alway return 0.
    */
    return 0;
}

static s32 internal_neta_dev_logblocknode_open(struct inode *i_inode, struct file *i_pfile)
{
    // ignore check it due to size is very small
    neta_lognodeprocessuser* pprocessuser = (neta_lognodeprocessuser*)kzalloc(sizeof(neta_lognodeprocessuser), GFP_KERNEL);
    mutex_init(&pprocessuser->mutexprocess);
    pprocessuser->bproducer = -1;
    pprocessuser->pprocessproducer = NULL;
    pprocessuser->pprocessconsumer = NULL;
    i_pfile->private_data = pprocessuser;
    return 0;
}

static s32 internal_neta_dev_logblocknode_release_producer(neta_lognodeprocessproducer* io_pproducer)
{
    s32 ret = 0;
    struct list_head *p = NULL, *n = NULL;
    neta_lognodethreadproducer* pthreadproducer = NULL;
    neta_logblock* plogblock;
    u32 referencenew;
    u32 bput;
    /*
    * Iterate every thread user in the processuser, atomic minus the reference and decide
    * whether need to free the correspondent log block here.
    */
    mutex_lock(&io_pproducer->mutexthreadproducerlist);
    list_for_each_safe(p, n, &io_pproducer->threadproducerlist) {
        pthreadproducer = list_entry(p, neta_lognodethreadproducer, threadproducernode);
        plogblock = pthreadproducer->plogblock;
        /*
        * The similar logic is in internal_neta_dev_logblocknode_ioctl_putlogblock.
        */
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
            referencenew = (u32)atomic_sub_return(1, (atomic_t*)&plogblock->reference.refcount);
            if (referencenew == 2) {
                ret = neta_logblock_tryput(plogblock, &bput);
                if (unlikely(ret < 0)) {
                    mutex_unlock(&io_pproducer->mutexthreadproducerlist);
                    return ret;
                }
                // do nothing here when bput is 0 or 1
            }
        }
        list_del(&pthreadproducer->threadproducernode);
        kfree(pthreadproducer);
    }
    mutex_unlock(&io_pproducer->mutexthreadproducerlist);
    /*
    * Sub the log writer process count.
    */
    atomic_sub_return(1, (atomic_t*)&__neta_log.writerprocesscount);
    /*
    * Delete mutex and free the neta_lognodeprocessproducer structure at last.
    */
    mutex_destroy(&io_pproducer->mutexthreadproducerlist);
    mutex_destroy(&io_pproducer->mutexgroupsharefd);
    kfree(io_pproducer);
    return 0;
}

static s32 internal_neta_dev_logblocknode_release_consumer(neta_lognodeprocessconsumer* io_pconsumer)
{
    s32 ret = 0;
    struct list_head *p = NULL, *n = NULL;
    neta_lognodethreadconsumer* pthreadconsumer = NULL;
    /*
    * Do not add the mutex outside the for, because the internal_neta_logblocknode_readerthreadquit
    * function will add mutex inside.
    * On the other hand, when do the release operation, it must be the process is been killed. No
    * other operation should be being done.
    */
    list_for_each_safe(p, n, &io_pconsumer->threadconsumerlist) {
        pthreadconsumer = list_entry(p, neta_lognodethreadconsumer, threadconsumernode);
        ret = internal_neta_logblocknode_readerthreadquit(pthreadconsumer, "release-readerthreadquit");
        if (unlikely(ret < 0)) {
            return ret;
        }
        list_del(&pthreadconsumer->threadconsumernode);
        kfree(pthreadconsumer);
    }
    /*
    * Delete mutex and free the neta_lognodeprocessconsumer structure at last.
    */
    mutex_destroy(&io_pconsumer->mutexthreadconsumerlist);
    mutex_destroy(&io_pconsumer->mutexgroupfd);
    kfree(io_pconsumer);
    return 0;
}

static s32 internal_neta_dev_logblocknode_release(struct inode *i_inode, struct file *i_pfile)
{
    neta_lognodeprocessuser* pprocessuser = (neta_lognodeprocessuser*)i_pfile->private_data;
    if (pprocessuser->bproducer == 1) {
        if (pprocessuser->pprocessproducer) {
            internal_neta_dev_logblocknode_release_producer(pprocessuser->pprocessproducer);
            pprocessuser->pprocessproducer = NULL;
        }
    }
    else if (pprocessuser->bproducer == 0) {
        if (pprocessuser->pprocessconsumer) {
            internal_neta_dev_logblocknode_release_consumer(pprocessuser->pprocessconsumer);
            pprocessuser->pprocessconsumer = NULL;
        }
    }
    mutex_destroy(&pprocessuser->mutexprocess);
    kfree(pprocessuser);
    return 0;
}

static const struct file_operations _logblocknode_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = internal_neta_dev_logblocknode_ioctl,
    .open = internal_neta_dev_logblocknode_open,
    .release = internal_neta_dev_logblocknode_release,

};

static char *netalog_devnode(struct device *i_pdev, umode_t *o_pmode)
{
    if (o_pmode != NULL) {
        *o_pmode = S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH;
    }
    return NULL;
}

s32 neta_dev_logblocknode_create()
{
    neta_logblockdevnode* dev = kzalloc(sizeof(neta_logblockdevnode), GFP_KERNEL);
    struct device* pdev;
    CHK_PTR_AND_RET_S32(dev, "kzalloc neta_logblockdevnode");
    CHK_SENTENCE_RET(alloc_chrdev_region(&dev->devno, 0, 1, NETA_LOGBASENODE_NAME), "alloc_chrdev_region");
    dev->majordevnum = MAJOR(dev->devno);
    cdev_init(&dev->cdev, &_logblocknode_fops);
    dev->cdev.owner = THIS_MODULE;
    CHK_SENTENCE_RET(cdev_add(&dev->cdev, dev->devno, 1), "cdev_add");
    dev->char_class = class_create(THIS_MODULE, NETA_LOGBASENODE_NAME);
    dev->char_class->devnode = netalog_devnode;
    pdev = device_create(dev->char_class, ((void *)0),
        dev->devno, ((void *)0), NETA_LOGBASENODE_NAME);
    CHK_PTR_AND_RET_S32(pdev, "device_create fail");
    __neta_node.lognode.plogblockdevnode = dev;
    return 0;
}

s32 neta_dev_logblocknode_destroy()
{
    device_destroy(__neta_node.lognode.plogblockdevnode->char_class, __neta_node.lognode.plogblockdevnode->devno);
    class_destroy(__neta_node.lognode.plogblockdevnode->char_class);
    cdev_del(&__neta_node.lognode.plogblockdevnode->cdev);
    unregister_chrdev_region(MKDEV(__neta_node.lognode.plogblockdevnode->majordevnum, 0), 1);
    kfree(__neta_node.lognode.plogblockdevnode);
    return 0;
}
