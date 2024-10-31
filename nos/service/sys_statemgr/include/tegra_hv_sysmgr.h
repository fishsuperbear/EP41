/*
 * Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
 *
 * This software is licensed under the terms of the GNU General Public
 * License version 2, as published by the Free Software Foundation, and
 * may be copied, distributed, and modified under those terms.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#ifndef _TEGRA_HV_SYSMGR_H
#define _TEGRA_HV_SYSMGR_H

#include <linux/types.h>

#define SYSMGR_IVCMSG_SIZE_MAX 64

enum hv_sysmgr_msg_type {
	HV_SYSMGR_MSG_TYPE_GUEST_EVENT		= 1,
	HV_SYSMGR_MSG_TYPE_VM_PM_CTL_CMD	= 2,
	HV_SYSMGR_MSG_TYPE_INVALID
};

enum hv_sysmgr_cmd_id {
	HV_SYSMGR_CMD_NORMAL_SHUTDOWN	= 0x0,
	HV_SYSMGR_CMD_NORMAL_REBOOT	= 0x1,
	HV_SYSMGR_CMD_NORMAL_SUSPEND	= 0x2,
	HV_SYSMGR_CMD_NORMAL_RESUME	= 0x3,
	HV_SYSMGR_CMD_INVALID		= 0xFFFFFFFF,
};

enum hv_sysmgr_resp_id {
	HV_SYSMGR_RESP_ACCEPTED		= 0x0,
	HV_SYSMGR_RESP_UNKNOWN_COMMAND	= 0xF,
};

/* This struct comes as payload of hv_pm_ctl_message */
struct __attribute((__packed__)) hv_sysmgr_command {
	uint32_t cmd_id;
	uint32_t resp_id;
};

struct __attribute((__packed__)) hv_sysmgr_message {
	/* msg class */
	uint32_t msg_type;
	/* id of open socket */
	uint32_t socket_id;
	/* client data area. Payload */
	uint8_t client_data[SYSMGR_IVCMSG_SIZE_MAX];
};

#endif /* _TEGRA_HV_SYSMGR_H */
