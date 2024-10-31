#ifndef DBG_CONF_H
#define DBG_CONF_H

#include <string>

#define DBG_VERSION "nos dbg v0.2 Release (2023-11-29)"
// #define DBG_OPT_MENU_LIST ""

#define TEGRA_TRIGGER_SYS_REBOOT "echo 1 > /sys/class/tegra_hv_pm_ctl/tegra_hv_pm_ctl/device/trigger_sys_reboot"
#define SSM_ZMQ_END_POINT "tcp://localhost:11155"

#endif