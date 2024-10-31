/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef NVTEGRAHV_YIELD_VCPU_H
#define NVTEGRAHV_YIELD_VCPU_H


#if defined(__cplusplus)
extern "C"
{
#endif

/**
 * @brief API to request Hypervisor to Yield VCPU to low priority VM
 *
 * @param[in] vm_id VM ID for low priority VM
 * @param[in] timeout_us Timer's timeout in microseconds
 *
 * @return
 * 		EOK: success\n
 * 		ENODEV Failed to open NvHv device node\n
 * 		ETIMEDOUT Timeout Error\n
 * 		EFAULT devctl call to NvHv device node returned failure\n
 *@pre
 *
 * @usage
 * - Allowed context for the API call
 *  - Interrupt handler: No
 *  - Signal handler: No
 *  - Thread-safe: Yes
 *  - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *  - Initialization: No
 *  - Run time: Yes
 *  - De-initialization: No
 *
 */
int32_t NvHvYieldVcpu(uint32_t vm_id, uint32_t timeout_us);

#if defined(__cplusplus)
}
#endif

#endif /* NVTEGRAHV_YIELD_VCPU_H */
