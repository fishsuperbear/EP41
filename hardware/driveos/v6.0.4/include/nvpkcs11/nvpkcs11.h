/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/**
 * @file
 * @brief NVIDIA PKCS #11 API header file
 * @details This header file contains the NVIDIA definitions for vendor-specific<br>
 * extensions that are for the PKCS11 interface provided for HPSE NVIDIA Tegra products.<br>
 * @version pkcs11 v3.0
 */

#ifndef NVPKCS11_H_
#include <stddef.h>
#define NVPKCS11_H_

#ifdef __cplusplus
extern "C" {
#endif
 //! @cond Doxygen_Suppress

/**
 * @defgroup nvpkcs11_macros Platform-specific Macros/Constants
 *
 * @details Defines PKCS11 platform-specific macro definitions for
 * HPSE NVIDIA Tegra products.
 *
 * @ingroup grp_pkcs11_api
 * @{
 */
#define CK_PTR *

#define CK_DECLARE_FUNCTION(returnType, name) \
   returnType name

#define CK_DECLARE_FUNCTION_POINTER(returnType, name) \
   returnType (* name)

#define CK_CALLBACK_FUNCTION(returnType, name) \
   returnType (* name)

#ifndef NULL_PTR
#define NULL_PTR NULL
#endif
/** @}*/
//! @endcond

/* Defines PKCS11 standard interface constants, data structures and function
 * prototype definitions for HPSE NVIDIA Tegra products.
 */
#include "pkcs11.h"
#include "nvpkcs11_future.h"


/**
 * @defgroup nvpkcs11_ext Vendor Extensions
 *
 * Defines PKCS11 vendor-specific interface extensions for HPSE NVIDIA Tegra products.
 *
 * @ingroup grp_pkcs11_api
 * @{
 */


/** Declaration of NV_CK_FUNCTION_LIST as a type */
typedef struct NV_CK_FUNCTION_LIST NV_CK_FUNCTION_LIST;
/** Declaration of NV_CK_FUNCTION_LIST_PTR as a type */
typedef NV_CK_FUNCTION_LIST CK_PTR NV_CK_FUNCTION_LIST_PTR;
/** Declaration of NV_CK_FUNCTION_LIST_PTR_PTR as a type */
typedef NV_CK_FUNCTION_LIST_PTR CK_PTR NV_CK_FUNCTION_LIST_PTR_PTR;


/**
 * @brief **C_NVIDIA_EncryptGetIV** gets the IV or CTR buffer data which was
 * generated during the AES encryption for CBC or CTR mode, respectively.
 *
 * @returns
 * - CKR_ARGUMENTS_BAD
 * - CKR_BUFFER_TOO_SMALL
 * - CKR_CRYPTOKI_NOT_INITIALIZED
 * - CKR_DATA_INVALID
 * - CKR_DATA_LEN_RANGE
 * - CKR_DEVICE_ERROR
 * - CKR_DEVICE_MEMORY
 * - CKR_FUNCTION_FAILED
 * - CKR_GENERAL_ERROR
 * - CKR_HOST_MEMORY
 * - CKR_OK
 * - CKR_OPERATION_NOT_INITIALIZED
 * - CKR_SESSION_CLOSED
 * - CKR_SESSION_HANDLE_INVALID
 * - CKR_USER_NOT_LOGGED_IN
 * - CKR_OPERATION_ACTIVE
 *
 * @param [in]  hSession The session handle for the encryption session initialized with **C_EncryptInit**
 * @param [in,out]  pIV Buffer for storing the IV or CTR data generated during the encryption session
 * @param [in,out]  pIVLen Pointer to the location that holds the length of the IV or CTR
 *
 * @pre This function shall be called after **C_Encrypt** or **C_EncryptFinal**.
 *
 * @note This function replaces the use of input IV or CTR in **C_EncryptInit**.
 *
 * @details This function is called as the last step in the encryption sequence,
 * and requires that first **C_EncryptInit** and then **C_Encrypt** or **C_EncryptInit** and then
 * one or more **C_EncryptUpdate(s)** followed by **C_EncryptFinal** have been called first.<br>
 * A call to **C_NVIDIA_EncryptGetIV** always terminates the active encryption unless it
 * returns CKR_BUFFER_TOO_SMALL, or is a successful call (returns CKR_OK)
 * to determine the length of the buffer needed to hold the data.<br>
 * If the function is successful it will return the IV value and the size of the IV value.<br>
 * **C_NVIDIA_EncryptGetIV** uses the convention described in Section 5.2 in the
 * PKCS #11 base documentation on producing output.
 *
 * @usage
 * - Allowed context for the API call
 *   - Thread-safe: Yes
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-init: No
 *
 **/
extern CK_RV C_NVIDIA_EncryptGetIV
(
	CK_SESSION_HANDLE hSession,
	CK_BYTE_PTR pIV,
	CK_BYTE_PTR pIVLen
);

/** C_EncryptGetIV is a legacy interface for C_NVIDIA_EncryptGetIV to allow for backwards compatibility. */
#define C_EncryptGetIV C_NVIDIA_EncryptGetIV

/** pointer to C_NVIDIA_EncryptGetIV */
typedef CK_RV (* CK_C_NVIDIA_EncryptGetIV)
(
	CK_SESSION_HANDLE hSession,
	CK_BYTE_PTR pIV,
	CK_BYTE_PTR pIVLen
);

/** NVIDIA functions list */
struct NV_CK_FUNCTION_LIST {
  CK_VERSION NV_version; /**< Cryptoki version */
  CK_C_NVIDIA_EncryptGetIV C_NVIDIA_EncryptGetIV; /**< Function to retrieve Initialization Vector (IV) after an encrypt operation */
};

/**
 * @brief The **CKM_NVIDIA_AES_CBC_KEY_DATA_WRAP** mechanism type should be used
 * with a CK_NVIDIA_AES_CBC_KEY_DATA_WRAP_PARAMS mechanism parameter to wrap either
 * one secret key or a pair of secret keys with custom data interleaved between
 * the two.
 *
 * @details This mechanism is intended for the C_WrapKey API. C_WrapKey's
 * third argument is the wrapping key (hWrappingKey) and the fourth argument
 * is the key to be wrapped (hKey).
 *
 * If hTrailingKey is CK_INVALID_HANDLE, the mechanism wraps a single key (data=[hKey])
 * using AES in CBC mode.
 *
 * If hTrailingKey is a valid handle, the mechanism wraps two keys with custom data
 * interleaved between them (data=[hKey|pData|hTrailingKey]) using AES in CBC mode.
 *
 * The mechanism uses CBC mode and generates a random IV that is returned to the caller
 * in the iv field of the mechanism parameter.
 *
 * The convention described in Section 5.2 of the PKCS #11 base documentation
 * can be used with C_WrapKey to compute the length of the wrapped key(s).
 *
 **/
#define CKM_NVIDIA_AES_CBC_KEY_DATA_WRAP (CKM_VENDOR_DEFINED | 0x00000001UL)

/**
* @brief **CK_NVIDIA_AES_CBC_KEY_DATA_WRAP_PARAMS** provides the parameters to
* the CKM_NVIDIA_AES_CBC_KEY_DATA_WRAP mechanism.
*/
typedef struct CK_NVIDIA_AES_CBC_KEY_DATA_WRAP_PARAMS {
   CK_BYTE_PTR      pData;         /**< Custom data pointer. Should be NULL if hTrailingKey is CK_INVALID_HANDLE. */
   CK_ULONG         ulLen;         /**< Custom data length in bytes. Should be a multiple of 16. Should be 0 if hTrailingKey is CK_INVALID_HANDLE. */
   CK_OBJECT_HANDLE hTrailingKey;  /**< Handle to the second key to be wrapped. */
   CK_BYTE          iv[16];        /**< Buffer to be overwritten with the IV generated for CBC mode. */
}  CK_NVIDIA_AES_CBC_KEY_DATA_WRAP_PARAMS;

/** Declaration of CK_AES_CBC_CUSTOM_DATA_WRAP_PARAMS_PTR as a type */
typedef CK_NVIDIA_AES_CBC_KEY_DATA_WRAP_PARAMS CK_PTR CK_NVIDIA_AES_CBC_KEY_DATA_WRAP_PARAMS_PTR;

/**
 * @brief The **CKM_NVIDIA_SP800_56C_TWO_STEPS_KDF** mechanism type should be used
 * with a CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS mechanism parameter to derive a CKK_AES
 * secret key from a CKK_AES or CKK_GENERIC_SECRET secret.
 *
 * @details The mechanism is intended for the C_DeriveKey API. The mechanism uses
 * two step key derivation as described in NISTSP800-56CREV.1: first extract
 * randomness from the base key and the salt, then expand it in counter mode with
 * an Info string.
 *
 * If applicable, the **L** field described in the NISTSP800-56CREV.1 standard should
 * be explicitly supplied as part of the Info string.
 **/
#define CKM_NVIDIA_SP800_56C_TWO_STEPS_KDF (CKM_VENDOR_DEFINED | 0x00000002UL)

/**
* @brief **CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS** provides the parameters to
* the CKM_NVIDIA_SP800_56C_TWO_STEPS_KDF mechanism.
*/
typedef struct CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS {
	CK_MECHANISM_TYPE prfType;   /**< Base MAC used in the underlying KDF operation. Only CKM_AES_CMAC supported for both AES and GENERIC_SECRET type base keys. */
	CK_BYTE_PTR       pSalt;     /**< Pointer to the salt used for the extract portion of the KDF. */
	CK_ULONG          ulSaltLen; /**< Length of the salt pointed to in pSalt. */
	CK_BYTE_PTR       pInfo;     /**< Info string for the expand stage of the KDF. */
	CK_ULONG          ulInfoLen; /**< Length of the info string pointed to by pInfo. Must be between 1 and 64 */
	CK_BYTE           ctr;       /**< Value of the counter for the expand stage of the KDF. */
}  CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS;

/** Declaration of CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS_PTR as a type */
typedef CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS CK_PTR CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS_PTR;

/** @}*/
#ifdef __cplusplus
}
#endif

#endif /* NVPKCS11_H_ */

