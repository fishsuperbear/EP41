#ifndef SYS_DEFINE_H
#define SYS_DEFINE_H

#include <linux/types.h>

#define SSM_CONFIG_FILE "/app/runtime_service/sys_statemgr/conf/ssm_config.json"
#define HV_PM_CTL_PATH	"/dev/tegra_hv_pm_ctl"

#define HV_PM_CTL_REBOOT "reboot"
#define HV_PM_CTL_SHUTDOWN "shutdown now"
#define HV_PM_CTL_SUSPEND "/bin/echo s2idle > /sys/power/mem_sleep && /bin/echo mem > /sys/power/state"

enum nos_sysmgr_cmd_id {
	NOS_SYSMGR_CMD_SOC_HEARTBEAT = 0x10,
    NOS_SYSMGR_CMD_SOC_REQUEST_RESET = 0x11,
    NOS_SYSMGR_CMD_SOC_REQUEST_RESTART = 0x12,
	NOS_SYSMGR_CMD_INVALID	 = 0xFF
};

enum class SocModeState : uint32_t {
    SOC_MODE_RUNNING = 0,
    SOC_MODE_STANDBY = 1,
    SOC_MODE_SHUTDOWN = 2,
    SOC_MODE_RESTART = 3,
    SOC_MODE_RESET = 4
};


#endif
