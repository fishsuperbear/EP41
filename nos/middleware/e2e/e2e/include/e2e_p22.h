#ifndef E2E_P22_H_
#define E2E_P22_H_
#include "e2e/e2e/include/e2e_types.h"
#include "e2e/e2e/include/e2e_sm.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Profile 22-header: Document ID 849: AUTOSAR_PRS_E2EProtocol P148
#define P22CRC_POS          0u
#define P22CRC_LEN          1u
#define P22COUNTER_POS      1u     //1.5
#define P22COUNTER_LEN      1u     //0.5
#define P22CALCULATE_CRC    Crc_CalculateCRC32P4

typedef struct {
    uint16_t  DataLength;
    uint8_t   DataIDList[16];
    uint8_t   MaxDeltaCounter;
    uint16_t  Offset;
} E2E_P22ConfigType;

typedef struct {
    uint8_t   Counter;
} E2E_P22ProtectStateType;

typedef enum {
    E2E_P22STATUS_OK            = 0x00u,
    E2E_P22STATUS_NONEWDATA     = 0x01u,
    E2E_P22STATUS_ERROR         = 0x07u,
    E2E_P22STATUS_REPEATED      = 0x08u,
    E2E_P22STATUS_OKSOMELOST    = 0x20u,
    E2E_P22STATUS_WRONGSEQUENCE = 0x40u
} E2E_P22CheckStatusType;

typedef struct {
    E2E_P22CheckStatusType Status;
    uint8_t Counter;
} E2E_P22CheckStateType;

Std_ReturnType E2E_P22Protect(const E2E_P22ConfigType* ConfigPtr, E2E_P22ProtectStateType* StatePtr,
                              uint8_t* DataPtr, uint16_t Length);
Std_ReturnType E2E_P22ProtectInit(E2E_P22ProtectStateType* StatePtr);                            
Std_ReturnType E2E_P22Forward(const E2E_P22ConfigType* ConfigPtr, E2E_P22ProtectStateType* StatePtr,
                              uint8_t* DataPtr, uint16_t Length, E2E_PCheckStatusType ForwardStatus);
Std_ReturnType E2E_P22Check(const E2E_P22ConfigType* ConfigPtr, E2E_P22CheckStateType* StatePtr,
                            const uint8_t* DataPtr, uint16_t Length);
Std_ReturnType E2E_P22CheckInit(E2E_P22CheckStateType* StatePtr);
E2E_PCheckStatusType E2E_P22MapStatusToSM(Std_ReturnType CheckReturn, E2E_P22CheckStatusType Status);
#ifdef __cplusplus
}
#endif
#endif