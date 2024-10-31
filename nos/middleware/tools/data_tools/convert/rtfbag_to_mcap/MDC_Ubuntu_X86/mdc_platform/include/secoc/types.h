/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: The declaration of the types of SecOC
 * Create: 2020-10-09
 */

#ifndef SECOC_TYPES_H
#define SECOC_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define KEY_ID_MAX_LENGTH 255

typedef enum {
    FVM_SUCCESS = 0,
    FVM_FAILED,
    FVM_BUSY,
    FVM_INVALID_PARAM
} FVMReturnType;

typedef struct {
    uint8_t *value;
    uint32_t length;
} Buff;

typedef struct {
    const uint8_t *value;
    uint32_t length;
} ConstBuff;

typedef FVMReturnType(*GetRxFreshness)(uint16_t freshnessValueID,
                                       uint16_t authVerifyAttempts,
                                       const ConstBuff *truncatedFreshnessValue,
                                       Buff *freshnessValue);

typedef FVMReturnType(*GetRxFreshnessAuthData)(uint16_t freshnessValueID,
                                               uint16_t authVerifyAttempts,
                                               const ConstBuff *truncatedFreshnessValue,
                                               const ConstBuff *authDataFreshnessValue,
                                               Buff *freshnessValue);

typedef FVMReturnType(*GetTxFreshness)(uint16_t freshnessValueID,
                                       Buff *freshnessValue);

typedef FVMReturnType(*GetTxFreshnessTruncData)(uint16_t freshnessValueID,
                                                Buff *freshnessValue,
                                                Buff *truncatedFreshnessValue);

typedef enum {
    SECOC_VERIFICATIONSUCCESS = 0,
    SECOC_VERIFICATIONFAILURE,
    SECOC_FRESHNESSFAILURE,
    SECOC_AUTHENTICATIONBUILDFAILURE,
    SECOC_NO_VERIFICATION
} VerificationResultType;

typedef struct {
    uint16_t freshnessValueId;
    VerificationResultType verificationResult;
    uint16_t dataId;
    uint64_t sequenceId;
} VerificationStatusType;

typedef void(*FVMVerificationStatusCallout)(VerificationStatusType verificationStatus);
typedef void(*FVMTxConfirmation)(uint16_t freshnessValueID);

typedef struct {
    GetRxFreshness getRxFreshness;
    GetRxFreshnessAuthData getRxFreshnessAuthData;
    GetTxFreshness getTxFreshness;
    GetTxFreshnessTruncData getTxFreshnessTruncData;
    FVMVerificationStatusCallout propagationVerificationStatus;
    FVMTxConfirmation txConfirmation;
} FVMOperateFunction;

typedef struct {
    uint16_t cacheSize;
    FVMOperateFunction fvmOpts;
} SecOCInitConfigType;

typedef enum {
    /* success result */
    SECOC_OK = 0,
    SECOC_FAILURE_SUCCESS,

    /* Error code for common */
    SECOC_ERROR = 0x1000,
    SECOC_OUT_OF_MEMORY,
    SECOC_MEMCPY_FAILED,

    /* Error code for secoc interface api */
    SECOC_UNINIT = 0x2000,
    SECOC_INVALID_PARAM,
    SECOC_CALLBACK_NULL,
    SECOC_DATA_LENGTH_INVALID,
    SECOC_CONFIG_AUTH_DATA_LEN_INVALID,
    SECOC_ENCRYPTION_INVALID_PARAM,

    /* Error code for config */
    SECOC_CONFIG_FV_LENGTH_INVALID = 0x3000,
    SECOC_CONFIG_FV_TRUNC_LENGTH_INVALID,
    SECOC_CONFIG_FV_AUTH_DATA_LENGTH_INVALID,
    SECOC_CONFIG_ALGO_MODE_INVALID,
    SECOC_CONFIG_ALGO_TRUNC_LENGTH_INVALID,
    SECOC_CONFIG_KEY_ID_LENGTH_INVALID,
    SECOC_CONFIG_ALIGN_MODE_INVALID,
    SECOC_CONFIG_PROPAGATION_MODE_INVAILD,
    SECOC_CONFIG_SECURED_AREA_INVAILD,

    /* Error code for data solve */
    SECOC_SOLVE_TX_DATA_LENGTH_INVALID = 0x4000,
    SECOC_SOLVE_RX_DATA_LENGTH_INVALID,
    SECOC_SOLVE_MAC_VERIFY_FAILED,
    SECOC_SOLVE_DEFAULT_FV_VERIFY_FAILED,
    SECOC_SOLVE_DEFAULT_MAC_VERIFY_FAILED,
    SECOC_SOLVE_FV_LEN_OVERFLOW,
    SECOC_SOLVE_GET_FV_PADDING_FAILED,
    SECOC_SOLVE_GET_MAC_PADDING_FAILED,
    SECOC_SOLVE_FVM_TX_ATTEMPT_MAX_TIMES,
    SECOC_SOLVE_FVM_RX_ATTEMPT_MAX_TIMES,

    /* Error code for fvm */
    SECOC_FVM_INVALID_PARAM = 0X5000,
    SECOC_FVM_ERROR,
    SECOC_FVM_BUSY,

    /* Error code for fvm adapter */
    SECOC_FVM_ADAPTER_REGISTER_OPS_INVALID = 0x6000,
    SECOC_FVM_ADAPTER_INVALID_PARAM,
    SECOC_FVM_ADAPTER_OPS_INVALID,

    /* Error code for encryption adapter */
    SECOC_SECMGR_ADAPTER_INVALID_PARAM = 0x7000,
    SECOC_SECMGR_ADAPTER_BEGIN_ERR,
    SECOC_SECMGR_ADAPTER_UPDATE_DATAID_ERR,
    SECOC_SECMGR_ADAPTER_UPDATE_PAYLOAD_ERR,
    SECOC_SECMGR_ADAPTER_FINISH_ERR,
    SECOC_SECMGR_ADAPTER_BUSY,
    SECOC_SECMGR_ADAPTER_ATTEMPT_MAX_TIMES,
    SECOC_SECMGR_ADAPTER_LOAD_LIB_ERR,
    SECOC_SECMGR_ADAPTER_RELEASE_LIB_ERR,
    SECOC_ENCRYPTION_ADAPTER_INVALID_PARAM,
    SECOC_ENCRYPTION_ADAPTER_OVERALL_ERR,
    SECOC_ENCRYPTION_ADAPTER_BEGIN_ERR,
    SECOC_ENCRYPTION_ADAPTER_UPDATE_ERR,
    SECOC_ENCRYPTION_ADAPTER_FINISH_ERR,
    SECOC_ENCRYPTION_ADAPTER_BUSY,
    SECOC_ENCRYPTION_ADAPTER_ATTEMPT_MAX_TIMES,

    /* Error code for secoc handler manager */
    SECOC_HANDLER_MGR_INVALID_HANDLER = 0x8000,

    /* Error code for queue handler manager */
    SECOC_QUEUE_MGR_ERROR = 0x9000,
    SECOC_QUEUE_MGR_UNINT,
    SECOC_QUEUE_MGR_QUEUE_EMPTY,
    SECOC_QUEUE_MGR_QUEUE_FULL
} SecOCReturnType;

typedef enum {
    GENERATE_SUCCESS = 0,
    GENERATE_FAILED,
    GENERATE_BUSY
} GenerateAuthenticatorResult;

typedef enum {
    SECOC_GEN_AUTHENTICATOR_SEGMENT = 0,
    SECOC_GEN_AUTHENTICATOR_OVERALL
} GenerateAuthenticatorType;

typedef enum {
    AUTH_HANDLER = 0,
    DEAUTH_HANDLER
} SecOCHandlerType;

typedef enum {
    AES_128_CMAC = 0,
    SIP_HASH_2_4
} AlgorithmMode;

typedef struct {
    AlgorithmMode mode;
    uint8_t keyIdBuff[KEY_ID_MAX_LENGTH];
    uint8_t keyIdBuffLength;
} SecOCAlgoConfig;

typedef GenerateAuthenticatorResult(*GenerateAuthenticator)(
    const SecOCAlgoConfig *algoConfig, const ConstBuff *calculateDataInfo, Buff* generatedDataInfo);

typedef GenerateAuthenticatorResult(*GenerateAuthenticatorBegin)(
    const SecOCAlgoConfig *algoConfig, const ConstBuff *calculateDataInfo, void** operationHandler);

typedef GenerateAuthenticatorResult(*GenerateAuthenticatorUpdate)(
    void** operationHandler, const ConstBuff *calculateDataInfo);

typedef GenerateAuthenticatorResult(*GenerateAuthenticatorFinish)(
    void** operationHandler, const ConstBuff *calculateDataInfo, Buff* generatedDataInfo);

typedef struct {
    GenerateAuthenticatorType type;
    GenerateAuthenticator generateAuthenticator;
    GenerateAuthenticatorBegin generateAuthenticatorBegin;
    GenerateAuthenticatorUpdate generateAuthenticatorUpdate;
    GenerateAuthenticatorFinish generateAuthenticatorFinish;
} AuthenticatorOperation;

typedef struct {
    bool enable;
    uint32_t offset;
    uint32_t length;
} SecOCSecuredArea;

typedef struct {
    bool enable;
    uint16_t offset; // start position in bits
    uint16_t length; // the length in bits
} SecOCUseAuthDataFreshness;

typedef struct {
    bool enable;
    uint8_t pattern;
} SecOCDefaultPattern;

typedef enum {
    BOTH = 0,
    FAILURE_ONLY,
    NONE
} SecOCVerificationStatusPropagationMode;

typedef struct {
    uint16_t buildAttempts; // For authentication and deauthentication
    uint16_t verifyAttempts; // Only for deauthentication
} SecOCAttempts;

typedef struct {
    uint32_t bufferLength; // Byte
    uint8_t *bufferPtr;    // The pointer points to buffer
    uint64_t sequenceId;
} DataInfo;

typedef struct {
    bool enable;
    uint16_t value;
} OptionalValue;

typedef enum {
    SECOC_1_BYTE_ALIGN = 0,
    SECOC_2_BYTE_ALIGN,
    SECOC_4_BYTE_ALIGN
} SecOCByteAlignmentType;

typedef struct SecOCHandler SecOCHandler;
typedef void(*SecOCCallback)(const SecOCHandler *handlerPtr, const SecOCReturnType result, const DataInfo* dataInfo);

#endif // SECOC_TYPES_H
