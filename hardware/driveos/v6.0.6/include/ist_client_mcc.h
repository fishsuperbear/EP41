/*
 * Copyright (c) 2022, NVIDIA Corporation. All Rights Reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation. Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation
 * is strictly prohibited.
 */
#ifndef __IST_CLIENT_MCC_H__
#define __IST_CLIENT_MCC_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

//==================================================================================================
// The following section defines the ISTClient callback functions that are registered with an MCC
// library and then get called by MCC lib to handle commands received from IST_Manager
//==================================================================================================

#define IST_CLIENT_MAX_CONFIGS (3U)

/**
 * ISTClient SetConfig function type
 *
 * @returns 0 on success
 */
typedef int (*ISTClient_SetConfig_fn_t)(
		void *ctx,                                      /**< [in] ISTClient_mcc_client_t::ctx */
		uint8_t num_configs,                            /**< [in] Number of IST config entries */
		const uint8_t configs[IST_CLIENT_MAX_CONFIGS] /**< [in] IST config entries */
	);

/**
 * ISTClient GetConfig function type
 *
 * @returns 0 on success
 */
typedef int (*ISTClient_GetConfig_fn_t)(
		void *ctx,                                /**< [in] ISTClient_mcc_client_t::ctx */
		uint8_t *num_configs,                     /**< [out] Number of IST config entries */
		uint8_t configs[IST_CLIENT_MAX_CONFIGS] /**< [out] IST config entries */
	);

/**
 * IST diagnostic result structure
 */
typedef struct ist_client_result {
	uint8_t hw_result;        /**< hw_result value (opaque to mcc) */
	uint8_t reserved_0;       /**< reserved_0 value (opaque to mcc) */
	uint8_t sw_rpl_status;    /**< sw_rpl_status value (opaque to mcc) */
	uint8_t sw_preist_status; /**< sw_preist_status value (opaque to mcc) */
} ist_client_result_t;

/**
 * ISTClient GetResult function type
 *
 * @returns 0 on success
 */
typedef int (*ISTClient_GetResult_fn_t)(
		void *ctx,                  /**< [in] ISTClient_mcc_client_t::ctx */
		uint8_t config,             /**< [out] IST config tested */
		ist_client_result_t *result /**< [out] IST diagnostic result */
	);

/**
 * Structure containing ISTClient callback functions which gets registered with MCC lib
 */
typedef struct {
	ISTClient_SetConfig_fn_t ISTClient_SetConfig; /**<  SetConfig callback */
	ISTClient_GetConfig_fn_t ISTClient_GetConfig; /**<  GetConfig callback */
	ISTClient_GetResult_fn_t ISTClient_GetResult; /**<  GetResult callback */

	void *ctx; /**< (optional) Context that is passed when calling ISTClient callbacks */
} ISTClient_mcc_client_t;

//==================================================================================================
// The following section defines the ISTClient MCC interface.
//
// ...Here's some pseudo'ish code to help illustrate the flow...
// [ist_client]
//
//	/* Initialize the MCC lib */
//	ISTClient_mcc_init(argc, argv);
//
//	/* Register ISTClient callbacks with MCC lib */
//	ISTClient_mcc_client_t ISTClient_cbs = {...};
//	ISTClient_mcc_register(&ISTClient_cbs);
//
//	/* Start communicating with IST Manager on MCU and process requests */
//	ISTClient_mcc_start();
//==================================================================================================

/**
 * Allocate and initialize MCC lib instance
 *
 * @param [in] argc: argc param passed to ist_client
 * @param [in] argv: argv param passed to ist_client - mcc lib opts follow '--mcc' param
 * @returns void * : On success, mcc_handle : On error, NULL
 */
extern void *ISTClient_mcc_init(int argc, char *argv[]);

/**
 * Deinitialize and cleanup MCC lib instance
 *
 * @param [in] mcc_handle: handle returned by ISTClient_mcc_init()
 * @returns 0 on success
 */
extern int ISTClient_mcc_deinit(void *mcc_handle);

/**
 * Register ISTClient callback functions
 *
 * @param [in] mcc_handle: handle returned by ISTClient_mcc_init()
 * @param [in] mcc_client: ist_client callbacks and context to be registered with mcc
 * @returns 0 on success
 */
extern int ISTClient_mcc_register(void *mcc_handle, ISTClient_mcc_client_t *mcc_client);

/**
 * Start ISTClient/ISTManager communication and process IST requests
 *
 * @param [in] mcc_handle: handle returned by ISTClient_mcc_init()
 * @returns 0 on success
 */
extern int ISTClient_mcc_start(void *mcc_handle);

#ifdef __cplusplus
}
#endif

#endif // __IST_CLIENT_MCC_H__
