#ifndef E2E_SM_H_
#define E2E_SM_H_
#include "e2e/e2e/include/e2e_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Document ID 849: AUTOSAR_PRS_E2EProtocol P79
typedef enum {
    E2E_P_OK = 0x00u,
    E2E_P_REPEATED = 0x01u,
    E2E_P_WRONGSEQUENCE = 0x02u,
    E2E_P_ERROR = 0x03u,
    E2E_P_NOTAVAILABLE = 0x04u,
    E2E_P_NONEWDATA = 0x05u,

    E2E_P_CHECKDISABLED = 0x06u,

    E2E_P_RESERVED = 0x07u,
} E2E_PCheckStatusType;

typedef enum {
    E2E_SM_VALID = 0x00u,
    E2E_SM_DEINIT = 0x01u,
    E2E_SM_NODATA = 0x02u,
    E2E_SM_INIT = 0x03u,
    E2E_SM_INVALID = 0x04u,

    E2E_SM_RESERVED = 0x07u,
} E2E_SMStateType;

typedef struct {                // restriction : Document ID 849: AUTOSAR_PRS_E2EProtocol P294
    uint8_t WindowSizeValid;    // WindowSizeValid >= 1 && minOkStateValid + maxErrorStateValid <= windowSizeValid
    uint8_t WindowSizeInvalid;  // WindowSizeInvalid <= WindowSizeValid && minOkStateInvalid + maxErrorStateInvalid <= WindowSizeInvalid
    uint8_t WindowSizeInit;     // windowSizeInit <= WindowSizeValid && minOkStateInit + maxErrorStateInit <= WindowSizeInit
    uint8_t MinOkStateInit;     // 1 <= minOkStateValid <= minOkStateInit <= minOkStateInvalid
    uint8_t MaxErrorStateInit;  // maxErrorStateValid >= maxErrorStateInit >= maxErrorStateInvalid >= 0
    uint8_t MinOkStateValid;
    uint8_t MaxErrorStateValid;
    uint8_t MinOkStateInvalid;
    uint8_t MaxErrorStateInvalid;
    boolean ClearToInvalid;
    boolean transitToInvalidExtended;
} E2E_SMConfigType;

typedef struct {
    uint8_t* ProfileStatusWindow;
    uint8_t WindowTopIndex;
    uint8_t OkCount;
    uint8_t ErrorCount;
    E2E_SMStateType SMState;
} E2E_SMCheckStateType;

Std_ReturnType E2E_SMCheck(E2E_PCheckStatusType ProfileStatus, const E2E_SMConfigType* ConfigPtr, E2E_SMCheckStateType* StatePtr);
Std_ReturnType E2E_SMCheckInit(E2E_SMCheckStateType* StatePtr, const E2E_SMConfigType* ConfigPtr);

uint8_t E2E_ChooseWindowSize(E2E_SMStateType CurrentState, const E2E_SMConfigType* ConfigPtr);
void E2E_SMAddStatus(E2E_PCheckStatusType ProfileStatus, const E2E_SMConfigType* ConfigPtr, E2E_SMCheckStateType* StatePtr);
void E2E_SMClearStatus(E2E_SMCheckStateType* StatePtr, const E2E_SMConfigType* ConfigPtr);
void E2E_SMClearRemainingStatus(E2E_SMCheckStateType* StatePtr, const E2E_SMConfigType* ConfigPtr, E2E_SMStateType NextState);
#ifdef __cplusplus
}
#endif
#endif