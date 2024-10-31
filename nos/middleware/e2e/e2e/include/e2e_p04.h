#ifndef E2E_P04_H_
#define E2E_P04_H_
#include "e2e/e2e/include/e2e_sm.h"
#include "e2e/e2e/include/e2e_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

// Profile 4-specific data : Document ID 849: AUTOSAR_PRS_E2EProtocol P83
#define P04LENGTH_POS 0u
#define P04LENGTH_LEN 2u
#define P04COUNTER_POS 2u
#define P04COUNTER_LEN 2u
#define P04DATAID_POS 4u
#define P04DATAID_LEN 4u
#define P04CRC_POS 8u
#define P04CRC_LEN 4u
#define P04CALCULATE_CRC Crc_CalculateCRC32P4

typedef struct {
    uint32_t DataID;
    uint16_t Offset;
    uint16_t MinDataLength;
    uint16_t MaxDataLength;
    uint16_t MaxDeltaCounter;
} E2E_P04ConfigType;

typedef struct {
    uint16_t Counter;
} E2E_P04ProtectStateType;

typedef enum {
    E2E_P04STATUS_OK = 0x00u,
    E2E_P04STATUS_NONEWDATA = 0x01u,
    E2E_P04STATUS_ERROR = 0x07u,
    E2E_P04STATUS_REPEATED = 0x08u,
    E2E_P04STATUS_OKSOMELOST = 0x20u,
    E2E_P04STATUS_WRONGSEQUENCE = 0x40u
} E2E_P04CheckStatusType;

typedef struct {
    E2E_P04CheckStatusType Status;
    uint16_t Counter;
} E2E_P04CheckStateType;

Std_ReturnType E2E_P04Protect(const E2E_P04ConfigType* ConfigPtr, E2E_P04ProtectStateType* StatePtr, uint8_t* DataPtr, uint16_t Length);
Std_ReturnType E2E_P04ProtectInit(E2E_P04ProtectStateType* StatePtr);
Std_ReturnType E2E_P04Forward(const E2E_P04ConfigType* ConfigPtr, E2E_P04ProtectStateType* StatePtr, uint8_t* DataPtr, uint16_t Length, E2E_PCheckStatusType ForwardStatus);
Std_ReturnType E2E_P04Check(const E2E_P04ConfigType* ConfigPtr, E2E_P04CheckStateType* StatePtr, const uint8_t* DataPtr, uint16_t Length);
Std_ReturnType E2E_P04CheckInit(E2E_P04CheckStateType* StatePtr);
E2E_PCheckStatusType E2E_P04MapStatusToSM(Std_ReturnType CheckReturn, E2E_P04CheckStatusType Status);
#ifdef __cplusplus
}
#endif
#endif