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
 * @defgroup nvtegrahv_yield_vcpu_api VCPU Yielding API
 * API to yield VCPUs to the low priority VMs.
 * @ingroup nvtegrahv_library
 * @{
 */

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
 *  - Re-entrant: Yes
 * - Required Privileges: "nvhv/yield_vcpu" custom ability
 * - API Group
 *  - Initialization: Yes
 *  - Run time: Yes
 *  - De-initialization: Yes
 *
 */
int32_t NvHvYieldVcpu(uint32_t vm_id, uint32_t timeout_us);

/**@} <!-- nvtegrahv_yield_vcpu_api VCPU Yielding --> */
#if defined(__cplusplus)
}
#endif

#endif /* NVTEGRAHV_YIELD_VCPU_H */
