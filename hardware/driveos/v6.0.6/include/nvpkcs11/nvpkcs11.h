/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

/** Declaration of CK_NVIDIA_CHANNEL_HANDLE as a type */
typedef CK_ULONG CK_NVIDIA_CHANNEL_HANDLE;
/** Declaration of CK_NVIDIA_CHANNEL_HANDLE_PTR as a type */
typedef CK_NVIDIA_CHANNEL_HANDLE CK_PTR CK_NVIDIA_CHANNEL_HANDLE_PTR;

/** Declaration of CK_NVIDIA_FLAGS as a type */
typedef CK_FLAGS CK_NVIDIA_FLAGS;
/** Declaration of CK_NVIDIA_FLAGS_PTR as a type */
typedef CK_NVIDIA_FLAGS CK_PTR CK_NVIDIA_FLAGS_PTR;

/** Declaration of NV_CK_FUNCTION_LIST as a type */
typedef struct NV_CK_FUNCTION_LIST NV_CK_FUNCTION_LIST;
/** Declaration of NV_CK_FUNCTION_LIST_PTR as a type */
typedef NV_CK_FUNCTION_LIST CK_PTR NV_CK_FUNCTION_LIST_PTR;
/** Declaration of NV_CK_FUNCTION_LIST_PTR_PTR as a type */
typedef NV_CK_FUNCTION_LIST_PTR CK_PTR NV_CK_FUNCTION_LIST_PTR_PTR;


/** Declaration of CKF_NVIDIA vendor extension flags */
/** Channel related flags */
#define CKF_NVIDIA_ZERO_COPY			(0x00000001UL) 	/**< Indicates this channel must be used with zero copy buffers */
#define CKF_NVIDIA_GCM_DECRYPT_UNAVAILABLE	(0x00000002UL)  /**< Indicates this channel does not support GCM decrypt operations */
/** TokenInfo extended flags  - these follow on from CKF_ERROR_STATE (0x01000000UL) */
#define CKF_NVIDIA_TOKEN_OK			(0x02000000UL)  /**< Token status is healthy */
#define CKF_NVIDIA_SECURE_STORAGE_FAILED	(0x04000000UL)  /**< This token does not have functional secure storage */
#define CKF_NVIDIA_SECURE_STORAGE_TAMPERED	(0x08000000UL)  /**< Secure storage may have been tampered, is not available */
#define CKF_NVIDIA_KEYLOAD_TIMEOUT		(0x10000000UL)  /**< It was not possible to transfer keys in a specified time, token keys will be unavailable */
#define CKF_NVIDIA_KEYLOAD_FAILED		(0x20000000UL)  /**< An error occurred when loading keys, token keys will be unavailable */
#define CKF_NVIDIA_TOKEN_ERROR			(0x40000000UL)  /**< An unspecified error occurred with the token */

/** Declaration of CKR_NVIDIA vendor extension return values */
#define CKR_NVIDIA_CHANNEL_NOT_FOUND		(CKR_VENDOR_DEFINED | 0x000000007UL)  /**< The requested channel could not be found */
#define CKR_NVIDIA_CHANNEL_CANNOT_OPEN		(CKR_VENDOR_DEFINED | 0x000000008UL)  /**< The requested channel could not be opened */
#define CKR_NVIDIA_SECURE_STORAGE_FAILED	(CKR_VENDOR_DEFINED | 0x000000009UL)  /**< This token does not have functional secure storage */
#define CKR_NVIDIA_SECURE_STORAGE_TAMPERED	(CKR_VENDOR_DEFINED | 0x000000010UL)  /**< Secure storage may have been tampered, is not available */

/** Declaration of CKA_NVIDIA vendor extension attributes */
#define CKA_NVIDIA_CALLER_NONCE (CKA_VENDOR_DEFINED | 0x00000001UL)

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

typedef struct CK_NVIDIA_CHANNEL_ATTRIBUTE {
	CK_FLAGS operations;                       /**< Details of operations are updated  */
	CK_NVIDIA_CHANNEL_HANDLE hChannel;         /**< Channel handle of the operations */
} CK_NVIDIA_CHANNEL_ATTRIBUTE;

typedef CK_NVIDIA_CHANNEL_ATTRIBUTE CK_PTR CK_NVIDIA_CHANNEL_ATTRIBUTE_PTR;


/**
 * @brief **C_NVIDIA_CommitTokenObjects** writes the current state of all token objects on a dynamic token to secure
 * storage. If there is a session open on any safety token in the system, then this function will fail with CKR_OPERATION_ACTIVE
 * in order to prevent any disruption to ongoing safety operations.
 *
 * @returns
 * - CKR_ARGUMENTS_BAD
 * - CKR_DEVICE_ERROR
 * - CKR_DEVICE_MEMORY
 * - CKR_FUNCTION_FAILED
 * - CKR_GENERAL_ERROR
 * - CKR_NVIDIA_SECURE_STORAGE_FAILED
 * - CKR_NVIDIA_SECURE_STORAGE_TAMPERED
 * - CKR_OK
 * - CKR_OPERATION_ACTIVE
 * - CKR_SESSION_CLOSED
 * - CKR_SESSION_HANDLE_INVALID
 * - CKR_SESSION_READ_ONLY_EXISTS
 * - CKR_SESSION_READ_ONLY
 * - CKR_TOKEN_WRITE_PROTECTED
 * - CKR_USER_NOT_LOGGED_IN
 *
 * @param [in]  hSession Previously obtained from **C_OpenSession** or **C_NVIDIA_OpenSession**
 * @param [in]  flags Currently not required, argument is reserved for future expansion
 *
 * @pre This function shall be called after **C_OpenSession** or **C_NVIDIA_OpenSession**.
 *
 * @note This function may take several minutes to complete and must only be called during the deinit phase and
 * may result in some PKCS11 operations being blocked while data is written.
 *
 * @details If no changes have been made, then this function will return CKR_OK.
 * If changes have been made, this function may take several minutes to complete.
 * This would apply to all tokens in the system, not just the token referenced in the call.
 * To protect safety-critical operations from being blocked, this function must only be called during
 * the deinit phase, as it could have an impact on live operations and boot time.
 * To enforce safe operation, this function will fail with CKR_OPERATION_ACTIVE if any application
 * has any safety token session open on this device (not just the token referenced in this call).
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
 *   - Runtime: No
 *   - De-init: Yes
 *
 **/
extern CK_RV C_NVIDIA_CommitTokenObjects
(
	CK_SESSION_HANDLE hSession,
	CK_FLAGS flags
);

/** pointer to C_NVIDIA_CommitTokenObjects */
typedef CK_RV (* CK_C_NVIDIA_CommitTokenObjects)
(
	CK_SESSION_HANDLE hSession,
	CK_FLAGS flags
);


/**
 * @brief **C_NVIDIA_InitializeChannel** opens a channel to a hardware engine.
 *
 * @returns
 * - CKR_ARGUMENTS_BAD
 * - CKR_CRYPTOKI_NOT_INITIALIZED
 * - CKR_DEVICE_ERROR
 * - CKR_DEVICE_MEMORY
 * - CKR_FUNCTION_FAILED
 * - CKR_GENERAL_ERROR
 * - CKR_HOST_MEMORY
 * - CKR_OK
 * - CKR_OPERATION_ACTIVE
 * - CKR_OPERATION_NOT_INITIALIZED
 * - CKR_NVIDIA_CHANNEL_NOT_FOUND
 * - CKR_NVIDIA_CHANNEL_CANNOT_OPEN
 *
 * @param [in]  ulChannelId Obtained from the device tree
 * @param [out]  phChannel Handle to be used with **C_NVIDIA_OpenSession**
 * @param [out]  pFlags Returns the flags that are associated with this channel, which could be none, or combinations
 * of CKF_NVIDIA_ZERO_COPY and CKF_NVIDIA_GCM_DECRYPT_UNAVAILABLE
 *
 * @details This is the first part of an extension to the PKCS#11 standard that allows targetting different hardware
 * engines. The handle can then be used with **C_NVIDIA_OpenSession** to create a session, or later with
 * **C_NVIDIA_FinalizeChannel** to close it.
 *
 * If the requested channel has already been opened, the same handle is returned.
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
 *   - Init: Yes
 *   - Runtime: No
 *   - De-init: No
 *
 **/
extern CK_RV C_NVIDIA_InitializeChannel
(
	CK_ULONG ulChannelId,
	CK_NVIDIA_CHANNEL_HANDLE_PTR phChannel,
	CK_NVIDIA_FLAGS_PTR pFlags
);

/** pointer to C_NVIDIA_InitializeChannel */
typedef CK_RV (* CK_C_NVIDIA_InitializeChannel)
(
	CK_ULONG ulChannelId,
	CK_NVIDIA_CHANNEL_HANDLE_PTR phChannel,
	CK_NVIDIA_FLAGS_PTR pFlags
);


/**
 * @brief **C_NVIDIA_OpenSession** opens a PKCS#11 session that can be configured to use different channels.
 *
 * @returns
 * - CKR_ARGUMENTS_BAD
 * - CKR_ATTRIBUTE_VALUE_INVALID
 * - CKR_BUFFER_TOO_SMALL
 * - CKR_CRYPTOKI_NOT_INITIALIZED
 * - CKR_DATA_INVALID
 * - CKR_DATA_LEN_RANGE
 * - CKR_DEVICE_ERROR
 * - CKR_DEVICE_MEMORY
 * - CKR_DEVICE_REMOVED
 * - CKR_FUNCTION_FAILED
 * - CKR_GENERAL_ERROR
 * - CKR_HOST_MEMORY
 * - CKR_OBJECT_HANDLE_INVALID
 * - CKR_OK
 * - CKR_OPERATION_ACTIVE
 * - CKR_OPERATION_NOT_INITIALIZED
 * - CKR_SESSION_CLOSED
 * - CKR_SESSION_COUNT
 * - CKR_SESSION_HANDLE_INVALID
 * - CKR_SESSION_PARALLEL_NOT_SUPPORTED
 * - CKR_SESSION_READ_ONLY_EXISTS
 * - CKR_SESSION_READ_WRITE_SO_EXISTS
 * - CKR_SLOT_ID_INVALID
 * - CKR_TOKEN_NOT_PRESENT
 * - CKR_TOKEN_NOT_RECOGNIZED
 * - CKR_TOKEN_WRITE_PROTECTED
 * - CKR_USER_NOT_LOGGED_IN
 *
 * @param [in]  slotID Same usage as **C_OpenSession**
 * @param [in]  flags Same usage as **C_OpenSession**
 * @param [in]  pApplication Same usage as **C_OpenSession**
 * @param [in]  Notify Same usage as **C_OpenSession**
 * @param [in,out] phSession Same usage as **C_OpenSession**
 * @param [in]  pChannelSettings Structure that contains mappings of operations to engines
 * @param [in]  ulChannelSettingsCount Number of entries in pChannelSettings structure
 * @param [in]  additionalFlags For future expansion, currently must be set to 0
 *
 * @details Extends the functionality of the standard C_OpenSession API call to allow channels to be configured
 * in that session. The pChannelSettings structure contains mappings of commands (e.g. CKF_ENCRYPT) to channel
 * handles (as obtained from **C_NVIDIA_InitializeChannel**). This allows a session to call (e.g.) C_Encrypt
 * and have that function target a different hardware engine queue (e.g. TZ-SE AES0).
 *
 * Multiple commands can share a channel, provided that the hardware engine type is usable for all operations.
 * Multiple mapping entries are possible, but commands must only be specified once.
 *
 * @usage
 * - Allowed context for the API call
 *   - Thread-safe: Yes
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: Yes
 * - API Group
 *   - Init: No
 *   - Runtime: Yes
 *   - De-init: No
 *
 **/

extern CK_RV C_NVIDIA_OpenSession
(
	CK_SLOT_ID slotID,
	CK_FLAGS flags,
	CK_VOID_PTR pApplication,
	CK_NOTIFY Notify,
	CK_SESSION_HANDLE_PTR phSession,
	CK_NVIDIA_CHANNEL_ATTRIBUTE_PTR pChannelSettings,
	CK_ULONG ulChannelSettingsCount,
	CK_NVIDIA_FLAGS additionalFlags
);

/** pointer to C_NVIDIA_OpenSession */
typedef CK_RV (* CK_C_NVIDIA_OpenSession)
(
	CK_SLOT_ID slotID,
	CK_FLAGS flags,
	CK_VOID_PTR pApplication,
	CK_NOTIFY Notify,
	CK_SESSION_HANDLE_PTR phSession,
	CK_NVIDIA_CHANNEL_ATTRIBUTE_PTR pChannelSettings,
	CK_ULONG ulChannelSettingsCount,
	CK_NVIDIA_FLAGS additionalFlags
);


/**
 * @brief **C_NVIDIA_FinalizeChannel** closes a channel handle if it is not in use.
 *
 * @returns
 * - CKR_ARGUMENTS_BAD
 * - CKR_CRYPTOKI_NOT_INITIALIZED
 * - CKR_FUNCTION_FAILED
 * - CKR_GENERAL_ERROR
 * - CKR_HOST_MEMORY
 * - CKR_OK
 * - CKR_OPERATION_ACTIVE
 * - CKR_OPERATION_NOT_INITIALIZED
 * - CKR_USER_NOT_LOGGED_IN
 * - CKR_NVIDIA_CHANNEL_NOT_FOUND
 * - CKR_NVIDIA_CHANNEL_CANNOT_OPEN
 *
 * @param [in]  hChannel
 *
 * @usage **C_NVIDIA_FinalizeChannel** can be called to close a channel when is not configured for use
 * in any session. If it is in use, CKR_OPERATION_ACTIVE is returned. This only needs to be called once
 * per handle, not once per **C_NVIDIA_InitializeChannel** call.
 *
 * - Allowed context for the API call
 *   - Thread-safe: Yes
 *   - Interrupt handler: No
 *   - Signal handler: No
 *   - Re-entrant: Yes
 *   - Async/Sync: Sync
 * - Required Privileges: None
 * - API Group
 *   - Init: No
 *   - Runtime: No
 *   - De-init: Yes
 *
 **/
extern CK_RV C_NVIDIA_FinalizeChannel
(
	CK_NVIDIA_CHANNEL_HANDLE hChannel
);

/** pointer to CK_C_NVIDIA_FinalizeChannel */
typedef CK_RV (* CK_C_NVIDIA_FinalizeChannel)
(
	CK_NVIDIA_CHANNEL_HANDLE hChannel
);


/** NVIDIA functions list */
struct NV_CK_FUNCTION_LIST {
	CK_VERSION NV_version; /**< Cryptoki version */
	CK_C_NVIDIA_EncryptGetIV C_NVIDIA_EncryptGetIV; /**< Function to retrieve Initialization Vector (IV) after an encrypt operation */
	CK_C_NVIDIA_CommitTokenObjects C_NVIDIA_CommitTokenObjects; /**< Function to write the current dynamic state of all token objects */
	CK_C_NVIDIA_InitializeChannel C_NVIDIA_InitializeChannel; /**< Function to open up a connection between the PKCS11 library and the HSM */
	CK_C_NVIDIA_OpenSession C_NVIDIA_OpenSession; /**< Function is an enhanced version of **C_OpenSession** with extra parameters */
	CK_C_NVIDIA_FinalizeChannel C_NVIDIA_FinalizeChannel; /**< Function closes the channel */
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
 * The wrapping key (hWrappingKey) can either be a session or a token object. The
 * keys to be wrapped (hKey and hTrailingKey) should not differ in their storage
 * attribute: they should both be session objects, or token objects.
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
} CK_NVIDIA_AES_CBC_KEY_DATA_WRAP_PARAMS;

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
} CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS;

/** Declaration of CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS_PTR as a type */
typedef CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS CK_PTR CK_NVIDIA_SP800_56C_TWO_STEPS_KDF_PARAMS_PTR;

/**
 * @brief The **CKM_NVIDIA_MACSEC_AES_KEY_WRAP** mechanism type should be used
 * with a CK_NVIDIA_MACSEC_AES_KEY_WRAP_PARAMS mechanism parameter to wrap or unwrap
 * a secret key.
 *
 * @details This mechanism is intended for the C_WrapKey and C_UnwrapKey API.
 *
 * It is designed to support NVIDIA MACsec hardware and software only.
 *
 **/
#define CKM_NVIDIA_MACSEC_AES_KEY_WRAP (CKM_VENDOR_DEFINED | 0x00000003UL)

/**
* @brief **CK_CKM_NVIDIA_MACSEC_AES_KEY_WRAP_PARAMS** provides the parameters to
* the CKM_NVIDIA_MACSEC_AES_KEY_WRAP mechanism.
*/
typedef struct CK_NVIDIA_MACSEC_AES_KEY_WRAP_PARAMS {
   CK_BYTE_PTR      pIv;                    /**< Pointer to the IV. */
   CK_ULONG         ulIvLen;                /**< Length of the IV in bytes. Should be 0 if pIv is NULL and 8 otherwise. */
   CK_BYTE_PTR      pMACsecMetadata;        /**< Custom data pointer. Should be NULL if used with C_WrapKey. */
   CK_ULONG         ulMACsecMetadataLen;    /**< Custom data length in bytes. Should be 0 if used with C_WrapKey. */
}  CK_NVIDIA_MACSEC_AES_KEY_WRAP_PARAMS;

/** Declaration of CK_NVIDIA_MACSEC_AES_KEY_WRAP_PARAMS_PTR as a type */
typedef CK_NVIDIA_MACSEC_AES_KEY_WRAP_PARAMS CK_PTR CK_NVIDIA_MACSEC_AES_KEY_WRAP_PARAMS_PTR;

/** @}*/
#ifdef __cplusplus
}
#endif

#endif /* NVPKCS11_H_ */

