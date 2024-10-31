#include "neta_node.h"

module_param(para_logmode, charp, S_IRUGO);

extern int  globalmem_init (void);
extern void  globalmem_exit(void);
extern int hallog_ctrl_init(void);
extern void hallog_ctrl_exit(void);

static int __init systemmanager_init(void)
{
    int ret;
    NETA_KLOG_INFO("neta systemmanager_init init! logmode is %s\n", para_logmode);
    NETA_KLOG_INFO("neta systemmanager.ko [build time]: %s %s.\n", __DATE__, __TIME__);

    globalmem_init();
    hallog_ctrl_init();

    if ((ret = neta_logblockgroup_driverinit())) {
        NETA_KLOG_ERR("neta neta_logblockgroup_driverinit ret=%d[%s]\n", ret, neta_kerr_info(ret));
        return ret;
    }
    if ((ret = neta_dev_logblocknode_create())) {
        NETA_KLOG_ERR("neta neta_logblockgroup_driverinit ret=%d[%s]\n", ret, neta_kerr_info(ret));
        return ret;
    }
    if ((ret = neta_proc_logbasenode_create()) < 0) {
        NETA_KLOG_ERR("neta neta_proc_logbasenode_create ret=%d[%s]\n", ret, neta_kerr_info(ret));
        return ret;
    }
    if ((ret = neta_proc_logglobalinfonode_create()) < 0) {
        NETA_KLOG_ERR("neta neta_proc_logglobalinfonode_create ret=%d[%s]\n", ret, neta_kerr_info(ret));
        return ret;
    }
    return 0;
}
module_init(systemmanager_init);

static void __exit systemmanager_exit(void)
{
    NETA_KLOG_INFO("neta systemmanager_init exit\n");

    /*
    * Set bquit 1 first.
    */
    __neta_log.bquit = 1;

    hallog_ctrl_exit();
    globalmem_exit();

    neta_proc_logglobalinfonode_destroy();
    neta_proc_logbasenode_destroy();

    neta_dev_logblocknode_destroy();

    /*
    * After destroy the proc node and the dev node, we then destroy the log node memory.
    */
    neta_logblockgroup_driverdeinit();
}
module_exit(systemmanager_exit);
MODULE_LICENSE("GPL");
