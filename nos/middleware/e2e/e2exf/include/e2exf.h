#ifndef E2EXF_TYPE_H_
#define E2EXF_TYPE_H_
#include <bits/stdint-uintn.h>

#include "e2e/e2e/include/e2e_p04.h"
#include "e2e/e2e/include/e2e_p22.h"
#include "e2e/e2e/include/e2e_sm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* [SWS_Xfrm_00032] ：A safety transformer shall return one of the errors shown in Table */
// #define E_OK (0x00U)
// #define E_SAFETY_INVALID_OK (0X40U)
// #define E_SAFETY_INVALID_REP (0x41U)
// #define E_SAFETY_INVALID_SEQ (0x42U)
// #define E_SAFETY_INVALID_ERR (0x43U)
// #define E_SAFETY_INVALID_NND (0x45U)
/* SoftError : */
#define E_SAFETY_SOFT_RUNTIMEERROR (0x77U)
#define E_SAFETY_SOFT_
/* HardError : */
#define E_SAFETY_HARD_RUNTIMEERROR (0xFFU)
#define E_SAFETY_HARD_

/* [SWS_E2EXf_00208] */
#define E_SAFETY_INVALID_CRC E_SAFETY_INVALID_ERR  //[DRAFT]

/* [SWS_Xfrm_00061] ：Development Errors */
#define E2EXF_E_UNINIT \
    (0x01U)                            // Error code if any other API service, except GetVersionInfo is called before the transformer
                                       // module was initialized with Init or after a call to DeInit
#define E2EXF_E_INIT_FAILED (0x02U)    // Error code if an invalid configuration set was selected
#define E2EXF_E_PARAM (0x03U)          // API service called with wrong parameter
#define E2EXF_E_PARAM_POINTER (0x04U)  // API service called with invalid pointer

typedef BOOL boolean;

typedef struct {
    uint32_t e2ecounter;
    uint32_t e2eSourceId;
} E2EXf_CSTransactionHandleType;

typedef enum { all16Bit = 0, alternating8Bit, lower12Bit, lower8Bit } DataIdModeEnum;

typedef enum {
    PRE_R4_2 = 0,  // Check has the legacy behavior, before AUTOSAR Release 4.2
    R4_2           // Check behaves like new P4/P5/P6 profiles introduced in AUTOSAR
                   // Release 4.2
} EndToEndProfileBehaviorEnum;

typedef struct {
    boolean transitToInvalidExtende;
} E2EProfileCompatibilityProps;

typedef enum {
    noTransformerStatusForwarding = 0,  //
    transformerStatusForwarding = 1     //
} DataTransformationStatusForwardingEnum;

typedef enum {
    PROFILE04 = 4U,
    PROFILE05 = 5U,
    PROFILE06 = 6U,
    PROFILE07 = 7U,
    PROFILE11 = 11U,
    PROFILE22 = 22U,
    PROFILE4M = 41U,
    PROFILE44 = 44U,

    PROFILE22_CUSTOM = 122U,

    UNDEFINE = 255U  //
} E2EXf_Profile;

/*  [SWS_Xfrm_00060] : This is the type of the data structure containing the initialization data for the transformer.*/
/* [SWS_E2EXf_00030] : Parent container for the configuration of E2E Transformer. The content is implementation-specific. */
typedef struct {
    uint32_t upperHeaderBitsToShift;
    uint32_t headerLength;
    boolean InPlace;
    boolean disableEndToEndCheck;
    boolean disableEndToEndStatemachine;

    DataTransformationStatusForwardingEnum DataTransformationStatusForwarding;
    E2EXf_Profile Profile;
    union {
        E2E_P04ConfigType Profile04;
        E2E_P22ConfigType Profile22;
    } ProfileConfig;

} E2EXf_ConfigType;

void E2EXf_Init(const E2EXf_ConfigType* config);

void E2EXf_P04_ProtectInit(E2E_P04ProtectStateType* stateptr);

void E2EXf_P04_CheckInit(E2E_P04CheckStateType* stateptr);

uint8_t E2EXf_P04(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P04ProtectStateType* state,
                  Std_TransformerForwardCode forwardedCode);

uint8_t E2EXf_Inv_P04(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P04CheckStateType* state,
                      const E2E_SMConfigType* smConfig, E2E_SMCheckStateType* smState);

void E2EXf_P22_ProtectInit(E2E_P22ProtectStateType* stateptr);

void E2EXf_P22_CheckInit(E2E_P22CheckStateType* stateptr);

uint8_t E2EXf_P22(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22ProtectStateType* state,
                  Std_TransformerForwardCode forwardedCode);

uint8_t E2EXf_Inv_P22(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22CheckStateType* state,
                      const E2E_SMConfigType* smConfig, E2E_SMCheckStateType* smState);

uint8_t E2EXf_P22_CUSTOM(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22ProtectStateType* state,
                         Std_TransformerForwardCode forwardedCode, void (*custom_before)(uint8_t*, uint32_t*), void (*custom_after)(uint8_t*, uint32_t*));

uint8_t E2EXf_Inv_P22_CUSTOM(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22CheckStateType* state,
                             const E2E_SMConfigType* smConfig, E2E_SMCheckStateType* smState, void (*custom_before)(uint8_t*, uint32_t*), void (*custom_after)(uint8_t*, uint32_t*));

#ifdef __cplusplus
}
#endif
#endif