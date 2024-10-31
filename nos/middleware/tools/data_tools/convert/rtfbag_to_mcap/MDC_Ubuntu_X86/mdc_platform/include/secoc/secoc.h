/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: The declaration of the public functions of SecOC
 * Create: 2020-10-09
 */

#ifndef SECOC_H
#define SECOC_H

#include "secoc/types.h"
#include "secoc/secoc_config_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize SecOC
 *
 * @param[in] config       The configuration for initializing SecOC
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType InitSecOC(const SecOCInitConfigType *config);

/**
 * @brief Initialize SecOC with Authenticator Opt
 *
 * @param[in] config             The configuration for initializing SecOC
 * @param[in] authenticatorOpts  Authenticator related Opts
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType InitSecOCWithAuthenticatorOpt(
    const SecOCInitConfigType *config, const AuthenticatorOperation authenticatorOpts);

/**
 * @brief Deinitialize SecOC
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType DeInitSecOC(void);

/**
 * @brief Create a SecOC Handler object
 *
 * @param[in]    config       The configuration for creating a SecOC Handler
 * @param[in]    handlerType  The type of handler that will be created
 * @param[out]   handler      The address of handler's pointer
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType CreateSecOCHandler(const SecOCConfigType *config, SecOCHandlerType handlerType, SecOCHandler **handler);

/**
 * @brief Destroy the specific SecOC handler
 *
 * @param[in] handler    The pointer of the handler will be destroyed
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType DestroySecOCHandler(const SecOCHandler *handler);

/**
 * @brief Get the SecOC Config object
 *
 * @param[in] handler   The pointer of the specific handler
 * @param[out] config   The pointer points to the cooresponding configuration information of the specific handler
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType GetSecOCConfig(const SecOCHandler *handler, SecOCConfigType *config);

/**
 * @brief Register the callback function of corresponding handler (only for asynchronous)
 *
 * @param[in] handler    The pointer points to SecOC handler
 * @param[in] callback   The pointer points to the function that needs to be called back after the authentication
 *                       field is added to the original user data.
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType RegisterSecOCCallback(const SecOCHandler *handler, SecOCCallback callback);

/**
 * @brief Cancel the registration of the callback function
 *        after adding the authentication field to the original user data (only for asynchronous)
 *
 * @param[in] handler  The pointer points to the SecOC handler
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType UnregisterSecOCCallback(const SecOCHandler *handler);

/**
 * @brief Get authentication data(freshness and mac) length which appending to user data
 * @param[in] handler The pointer points to the SecOC handler
 * @param[out] appendLength The pointer points to authentication data(freshness and mac) length
 *
 * @return SecOCReturnType SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType GetAppendAuthDataLength(const SecOCHandler *handler, uint32_t *appendLength);

/**
 * @brief Start adding authentication fields to original user data (Reentrant & Asynchronous)
 *
 * @param[in] handler  The pointer points to the SecOC handler
 * @param[in] userDataInfo  The pointer points to the user data information
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType StartGenerateSecOCAuthData(const SecOCHandler *handler, const DataInfo *userDataInfo);

/**
 * @brief Get the SecOC Authentication data
 *        (The same handle is not reentrant, but an unreasonable Handle can be reentrant & Synchronous)
 *
 * @param[in]    handler   The pointer points to the SecOC handler
 * @param[inout] userDataInfo  The pointer points to the user data information
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType GetSecOCAuthData(const SecOCHandler *handler, const DataInfo *userDataInfo);

/**
 * @brief Start deauthentication of authenticated user data (Reentrant & Asynchronous)
 *
 * @param[in] handler   The pointer points to the SecOC handler
 * @param[in] userDataInfo  The pointer points to the user data information
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType StartCheckSecOCAuthData(const SecOCHandler *handler, const DataInfo *userDataInfo);

/**
 * @brief Obtain the original user data after deauthentication
 *        (The same handle is not reentrant, but an unreasonable Handle can be reentrant & Synchronous)
 *
 * @param[in]    handler   The pointer points to the SecOC handler
 * @param[inout] userDataInfo  The pointer points to the user data information
 *
 * @return SecOCReturnType   SECOC_OK if success, otherwise return the error code.
 */
SecOCReturnType CheckSecOCAuthData(const SecOCHandler *handler, DataInfo *userDataInfo);

#ifdef __cplusplus
}
#endif

#endif // SECOC_H
