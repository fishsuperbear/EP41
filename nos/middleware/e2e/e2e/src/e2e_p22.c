#include "e2e/e2e/include/e2e_p22.h"
#include "crc/include/crc.h"

/* P22 Header Offset :
   CRC     : 1
   Counter : 0      */
Std_ReturnType E2E_P22Protect(const E2E_P22ConfigType* ConfigPtr, E2E_P22ProtectStateType* StatePtr,
                              uint8_t* DataPtr, uint16_t Length) {
    uint8_t  Counter    = 0 , ComputedCRC    = 0;                           
    uint16_t Offset     = ConfigPtr->Offset >> 3;
    //1. Verify inputs of the protect function
    if(!(ConfigPtr != NULL && StatePtr != NULL && DataPtr != NULL))
        return E2E_E_INPUTERR_NULL;
    if(!(Length == ConfigPtr->DataLength/8 && Length >= Offset + 2)) 
        return E2E_E_INPUTERR_WRONG;
    //2. increment Counter
    StatePtr->Counter = (StatePtr->Counter + 1) % 16;
    //3. Write the Counter
    Counter = StatePtr->Counter;
    *(DataPtr + Offset + P22COUNTER_POS) = (*(DataPtr + Offset + P22COUNTER_POS) & 0xF0) | (Counter & 0x0F); 
    //4. Compute CRC  PS:AUTOSAR_PRS_E2EProtocol P152 
    if(Offset > 0) {
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[0], Offset, 0xFF, TRUE);
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[Offset + P22CRC_LEN], Length - Offset - 1, ComputedCRC, FALSE);
    } else {
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[1], Length - 1, 0xFF, TRUE);
    }
    ComputedCRC = Crc_CalculateCRC8H2F(&ConfigPtr->DataIDList[Counter], 1, ComputedCRC, FALSE);
    //5. Write CRC
    *(DataPtr + Offset) = ComputedCRC & 0xFF;

    return E2E_E_OK;
}

Std_ReturnType E2E_P22Forward(const E2E_P22ConfigType* ConfigPtr, E2E_P22ProtectStateType* StatePtr,
                              uint8_t* DataPtr, uint16_t Length, E2E_PCheckStatusType ForwardStatus) {
    uint8_t  Counter    = 0 , ComputedCRC    = 0;                           
    uint16_t Offset     = ConfigPtr->Offset >> 3;
    uint8_t   WrongDataIDList[16];
    for(uint32_t i = 0; i < 16; i++)
        WrongDataIDList[i] = ConfigPtr->DataIDList[i] + 1;
    //1. Verify inputs of the protect function
    if(!(ConfigPtr != NULL && StatePtr != NULL && DataPtr != NULL))
        return E2E_E_INPUTERR_NULL;
    if(!(Length == ConfigPtr->DataLength/8 && Length >= Offset + 2 && ForwardStatus != E2E_P_NONEWDATA)) 
        return E2E_E_INPUTERR_WRONG;
    //2. increment Counter
    if(ForwardStatus == E2E_P_WRONGSEQUENCE)
        StatePtr->Counter = StatePtr->Counter + ConfigPtr->MaxDeltaCounter + 1;
    else if(ForwardStatus != E2E_P_REPEATED)
        StatePtr->Counter++;
    StatePtr->Counter %= 16;
    //3. Write the Counter
    Counter = StatePtr->Counter;
    *(DataPtr + Offset + P22COUNTER_POS) = (*(DataPtr + Offset + P22COUNTER_POS) & 0xF0) | (Counter & 0x0F); 
    //4. Compute CRC  PS:AUTOSAR_PRS_E2EProtocol P152 
    if(Offset > 0) {
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[0], Offset, 0xFF, TRUE);
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[Offset + P22CRC_LEN], Length - Offset - 1, ComputedCRC, FALSE);
    } else {
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[1], Length - 1, 0xFF, TRUE);
    }
    if(ForwardStatus == E2E_P_ERROR)
        ComputedCRC = Crc_CalculateCRC8H2F(&WrongDataIDList[Counter], 1, ComputedCRC, FALSE);
    else
        ComputedCRC = Crc_CalculateCRC8H2F(&ConfigPtr->DataIDList[Counter], 1, ComputedCRC, FALSE);
    //5. Write CRC
    *(DataPtr + Offset) = ComputedCRC & 0xFF;

    return E2E_E_OK;
}

Std_ReturnType E2E_P22Check(const E2E_P22ConfigType* ConfigPtr, E2E_P22CheckStateType* StatePtr,
                            const uint8_t* DataPtr, uint16_t Length) {
    boolean NewDataAvailable   = FALSE;
    uint8_t  Recv_Counter   = 0, Recv_CRC   = 0, ComputedCRC   = 0,
             DeltaCounter   = 0;
    uint16_t Offset         = ConfigPtr->Offset >> 3;
    // 1. Verify inputs of the protect function
    do {
        if(!(ConfigPtr != NULL && StatePtr != 0))
            return E2E_E_INPUTERR_NULL;
        if(!((DataPtr != NULL && Length != 0) || (DataPtr == NULL && Length == 0)))
            return E2E_E_INPUTERR_WRONG;
        if(!(DataPtr != NULL))
            break;
        if(!(Length == ConfigPtr->DataLength/8 && Length >= Offset+2))
            return E2E_E_INPUTERR_WRONG;
    NewDataAvailable = TRUE;
    } while(0);
    // 1#.NewDataAvailable Judge , false jump to check
    do {
    if(NewDataAvailable == FALSE)
        break;
    // 2. Read Counter
    Recv_Counter = *(DataPtr + Offset + P22COUNTER_POS) & 0x0F;
    // 3. Read CRC
    Recv_CRC = *(DataPtr + Offset + P22CRC_POS) & 0xFF;
    // 4. Compute CRC
    if(Offset > 0) {
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[0], Offset, 0xFF, TRUE);
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[Offset + P22CRC_LEN], Length - Offset - 1, ComputedCRC, FALSE);
    } else {
        ComputedCRC = Crc_CalculateCRC8H2F(&DataPtr[1], Length - 1, 0xFF, TRUE);
    }
    ComputedCRC = Crc_CalculateCRC8H2F(&ConfigPtr->DataIDList[Recv_Counter], 1, ComputedCRC, FALSE);
    } while(0);
    // 5. Do checks
    do {
        if(!(NewDataAvailable == TRUE)) {
            StatePtr->Status = E2E_P22STATUS_NONEWDATA;
            break;
        }
        if(!(Recv_CRC == ComputedCRC)) {
            StatePtr->Status = E2E_P22STATUS_ERROR;
            break;
        }
    // 5#. Compute the DeltaCounter {
        if (Recv_Counter >= StatePtr->Counter)
            DeltaCounter = Recv_Counter - StatePtr->Counter;
        else 
            DeltaCounter = 0xF - StatePtr->Counter + Recv_Counter + 1;
    //}
        if(!(DeltaCounter <= ConfigPtr->MaxDeltaCounter && DeltaCounter >= 0))
            StatePtr->Status = E2E_P22STATUS_WRONGSEQUENCE;
        else if(!(DeltaCounter > 0))
            StatePtr->Status = E2E_P22STATUS_REPEATED;
        else if(!(DeltaCounter == 1))
            StatePtr->Status = E2E_P22STATUS_OKSOMELOST;
        else 
            StatePtr->Status = E2E_P22STATUS_OK;
        StatePtr->Counter = Recv_Counter;
    } while(0);
    return E2E_E_OK;
}

E2E_PCheckStatusType E2E_P22MapStatusToSM(Std_ReturnType CheckReturn,
                                          E2E_P22CheckStatusType Status) {
    if(CheckReturn != E2E_E_OK) 
        return E2E_P_ERROR;
    switch (Status)
    {
    case E2E_P22STATUS_OK:
        return E2E_P_OK;
        break;
    case E2E_P22STATUS_OKSOMELOST:
        return E2E_P_OK;
        break;
    case E2E_P22STATUS_ERROR:
        return E2E_P_ERROR;
        break;
    case E2E_P22STATUS_REPEATED:
        return E2E_P_REPEATED;
        break;
    case E2E_P22STATUS_NONEWDATA:
        return E2E_P_NONEWDATA;
        break;
    case E2E_P22STATUS_WRONGSEQUENCE:
        return E2E_P_WRONGSEQUENCE;
        break;
    default:
        break;
    }
    return E2E_P_RESERVED;        
}

Std_ReturnType E2E_P22ProtectInit(E2E_P22ProtectStateType* StatePtr) {
    if(StatePtr == NULL)
        return E2E_E_INPUTERR_NULL;
    StatePtr->Counter = 0;
    return E2E_E_OK;
}

Std_ReturnType E2E_P22CheckInit(E2E_P22CheckStateType* StatePtr) {
    if(StatePtr == NULL)
        return E2E_E_INPUTERR_NULL;
    StatePtr->Counter = 0xF;
    StatePtr->Status  = E2E_P22STATUS_ERROR;
    return E2E_E_OK;
}