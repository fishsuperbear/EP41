#include "e2e/e2e/include/e2e_p04.h"
#include "crc/include/crc.h"

/* P04 Header Offset :
   Length  : 0
   Counter : 2
   Data ID : 4
   CRC     : 8      */
Std_ReturnType E2E_P04Protect(const E2E_P04ConfigType *ConfigPtr,
                              E2E_P04ProtectStateType *StatePtr, uint8_t *DataPtr,
                              uint16_t Length) {
    uint16_t Offset      = 0;                            
    uint32_t ComputedCRC = 0xFFFFFFFF;
    // 1.Verify inputs of the protect function
    if(!(ConfigPtr != NULL && StatePtr != NULL && DataPtr != NULL))
        return E2E_E_INPUTERR_NULL;
    if(!(Length>=ConfigPtr->MinDataLength/8 && Length<=ConfigPtr->MaxDataLength/8))
        return E2E_E_INPUTERR_WRONG;
    // 2.Compute offset
    Offset = ConfigPtr->Offset >> 3;
    // 3.Write Length to Dataptr
    *(DataPtr + Offset    ) = (Length >> 8) & 0xFF;
    *(DataPtr + Offset + 1) =  Length       & 0xFF;
    // 4.Write Counter
    *(DataPtr + Offset + 2) = (StatePtr->Counter >> 8) & 0xFF;
    *(DataPtr + Offset + 3) =  StatePtr->Counter       & 0xFF;
    // 5.Write DataID
    *(DataPtr + Offset + 4) = (ConfigPtr->DataID >> 24) & 0xFF;
    *(DataPtr + Offset + 5) = (ConfigPtr->DataID >> 16) & 0xFF;
    *(DataPtr + Offset + 6) = (ConfigPtr->DataID >> 8 ) & 0xFF;
    *(DataPtr + Offset + 7) =  ConfigPtr->DataID        & 0xFF;
    // 6.Compute CRC
    ComputedCRC = Crc_CalculateCRC32P4(&DataPtr[0], Offset + P04CRC_POS, ComputedCRC, TRUE);
    if(Offset + P04CRC_POS + P04CRC_LEN < Length)
        ComputedCRC = Crc_CalculateCRC32P4(&DataPtr[Offset + P04CRC_POS + P04CRC_LEN], Length - (Offset + P04CRC_POS + P04CRC_LEN), ComputedCRC, FALSE);
    // 7.Write CRC
    *(DataPtr + Offset + 8 ) = (ComputedCRC >> 24) & 0xFF;
    *(DataPtr + Offset + 9 ) = (ComputedCRC >> 16) & 0xFF;
    *(DataPtr + Offset + 10) = (ComputedCRC >> 8 ) & 0xFF;
    *(DataPtr + Offset + 11) =  ComputedCRC        & 0xFF;
    // 8.Increment Counter
    if(StatePtr->Counter == 0xFFFF) 
        StatePtr->Counter = 0;
    else 
        StatePtr->Counter++;
    return E2E_E_OK;
}

Std_ReturnType E2E_P04Forward(const E2E_P04ConfigType *ConfigPtr,
                              E2E_P04ProtectStateType *StatePtr, uint8_t *DataPtr,
                              uint16_t Length, E2E_PCheckStatusType ForwardStatus) {
    uint16_t Offset      = 0;                            
    uint32_t ComputedCRC = 0xFFFFFFFF;
    // 1.Verify inputs of the protect function
    if(!(ConfigPtr != NULL && StatePtr != NULL && DataPtr != NULL))
        return E2E_E_INPUTERR_NULL;
    if(!(Length>=ConfigPtr->MinDataLength/8 && Length<=ConfigPtr->MaxDataLength/8 && ForwardStatus != E2E_P_NONEWDATA))
        return E2E_E_INPUTERR_WRONG;
    // 2.Compute offset
    Offset = ConfigPtr->Offset >> 3;
    // 3.Write Length to Dataptr
    *(DataPtr + Offset    ) = (Length >> 8) & 0xFF;
    *(DataPtr + Offset + 1) =  Length       & 0xFF;
    // 4.Write Counter
    if(ForwardStatus == E2E_P_REPEATED)
        StatePtr->Counter--;
    else if(ForwardStatus == E2E_P_WRONGSEQUENCE)
        StatePtr->Counter = StatePtr->Counter + ConfigPtr->MaxDeltaCounter;
    *(DataPtr + Offset + 2) = (StatePtr->Counter >> 8) & 0xFF;
    *(DataPtr + Offset + 3) =  StatePtr->Counter       & 0xFF;
    // 5.Write DataID
    if(ForwardStatus == E2E_P_ERROR) {
        *(DataPtr + Offset + 4) = ((ConfigPtr->DataID + 1) >> 24) & 0xFF;
        *(DataPtr + Offset + 5) = ((ConfigPtr->DataID + 1) >> 16) & 0xFF;
        *(DataPtr + Offset + 6) = ((ConfigPtr->DataID + 1) >> 8 ) & 0xFF;
        *(DataPtr + Offset + 7) =  (ConfigPtr->DataID + 1)        & 0xFF;
    } else {
        *(DataPtr + Offset + 4) = (ConfigPtr->DataID >> 24) & 0xFF;
        *(DataPtr + Offset + 5) = (ConfigPtr->DataID >> 16) & 0xFF;
        *(DataPtr + Offset + 6) = (ConfigPtr->DataID >> 8 ) & 0xFF;
        *(DataPtr + Offset + 7) =  ConfigPtr->DataID        & 0xFF;
    }
    // 6.Compute CRC
    ComputedCRC = Crc_CalculateCRC32P4(&DataPtr[0], Offset + P04CRC_POS, ComputedCRC, TRUE);
    if(Offset + P04CRC_POS + P04CRC_LEN < Length)
        ComputedCRC = Crc_CalculateCRC32P4(&DataPtr[Offset + P04CRC_POS + P04CRC_LEN], Length - (Offset + P04CRC_POS + P04CRC_LEN), ComputedCRC, FALSE);
    // 7.Write CRC
    *(DataPtr + Offset + 8 ) = (ComputedCRC >> 24) & 0xFF;
    *(DataPtr + Offset + 9 ) = (ComputedCRC >> 16) & 0xFF;
    *(DataPtr + Offset + 10) = (ComputedCRC >> 8 ) & 0xFF;
    *(DataPtr + Offset + 11) =  ComputedCRC        & 0xFF;
    // 8.Increment Counter
    if(StatePtr->Counter == 0xFFFF) 
        StatePtr->Counter = 0;
    else 
        StatePtr->Counter++;
    return E2E_E_OK;
}

Std_ReturnType E2E_P04Check(const E2E_P04ConfigType *ConfigPtr, 
                            E2E_P04CheckStateType *StatePtr, const uint8_t *DataPtr, 
                            uint16_t Length) {
    // 1.Verify inputs of the check function
    boolean NewDataAvailable = FALSE;
    uint16_t Recv_Length = 0, Recv_Counter = 0, DeltaCounter = 0         , Offset = 0;
    uint32_t Recv_DataID = 0, Recv_CRC     = 0, ComputedCRC  = 0xFFFFFFFF;
    do {
    if(!(ConfigPtr != NULL && StatePtr != NULL))
        return E2E_E_INPUTERR_NULL;
    if(!((DataPtr != NULL && Length != 0) || (DataPtr == NULL && Length == 0)))
        return E2E_E_INPUTERR_WRONG;
    if(!(DataPtr != NULL))
        break;
    if(!((Length >= ConfigPtr->MinDataLength/8) && (Length <= ConfigPtr->MaxDataLength/8)))
        return E2E_E_INPUTERR_WRONG;
    NewDataAvailable = TRUE;
    } while(0);
    // 1#.NewDataAvailable Judge , false jump to check 
    do {
    if(NewDataAvailable == FALSE)
        break;
    // 2.Compute offset
    Offset = ConfigPtr->Offset/8;
    // 3.Read Length
    Recv_Length  = (*(DataPtr + Offset + P04LENGTH_POS ) << 8) + (*(DataPtr + Offset + P04LENGTH_POS  + 1));
    // 4.Read Counter
    Recv_Counter = (*(DataPtr + Offset + P04COUNTER_POS) << 8) + (*(DataPtr + Offset + P04COUNTER_POS + 1));
    // 5.Read DataID
    Recv_DataID  = (*(DataPtr + Offset + P04DATAID_POS    ) << 24) + (*(DataPtr + Offset + P04DATAID_POS + 1) << 16)
                 + (*(DataPtr + Offset + P04DATAID_POS + 2) << 8 ) + (*(DataPtr + Offset + P04DATAID_POS + 3)      );
    // 6.Read CRC
    Recv_CRC = (*(DataPtr + Offset + P04CRC_POS    ) << 24) + (*(DataPtr + Offset + P04CRC_POS + 1) << 16)
             + (*(DataPtr + Offset + P04CRC_POS + 2) << 8 ) + (*(DataPtr + Offset + P04CRC_POS + 3)      );
    // 7.Compute CRC
    ComputedCRC = Crc_CalculateCRC32P4(&DataPtr[0], Offset + P04CRC_POS, ComputedCRC, TRUE);
    if(Offset + P04CRC_POS + P04CRC_LEN < Length)
        ComputedCRC = Crc_CalculateCRC32P4(&DataPtr[Offset + P04CRC_POS + P04CRC_LEN], Length - (Offset + P04CRC_POS + P04CRC_LEN), ComputedCRC, FALSE);
    } while(0);
    // 8.Do checks
    do {
        if(!(NewDataAvailable == TRUE)) {
            StatePtr->Status = E2E_P04STATUS_NONEWDATA;
            break;
        }
        if(!(Recv_CRC == ComputedCRC && Recv_DataID == ConfigPtr->DataID && Recv_Length == Length)) {
            StatePtr->Status = E2E_P04STATUS_ERROR;
            break;
        }
    // 8#. Compute the DeltaCounter {
        if (Recv_Counter >= StatePtr->Counter)
            DeltaCounter = Recv_Counter - StatePtr->Counter;
        else 
            DeltaCounter = 0xFFFF - StatePtr->Counter + Recv_Counter + 1;
    //}
        if(!(DeltaCounter <= ConfigPtr->MaxDeltaCounter && DeltaCounter >= 0))
            StatePtr->Status = E2E_P04STATUS_WRONGSEQUENCE;
        else if(!(DeltaCounter > 0))
            StatePtr->Status = E2E_P04STATUS_REPEATED;
        else if(!(DeltaCounter == 1))
            StatePtr->Status = E2E_P04STATUS_OKSOMELOST;
        else 
            StatePtr->Status = E2E_P04STATUS_OK;
        StatePtr->Counter = Recv_Counter;

    } while(0);
    return E2E_E_OK;
}

E2E_PCheckStatusType E2E_P04MapStatusToSM(Std_ReturnType CheckReturn,
                                          E2E_P04CheckStatusType Status) {
    if(CheckReturn != E2E_E_OK) 
        return E2E_P_ERROR;
    switch (Status)
    {
    case E2E_P04STATUS_OK:
        return E2E_P_OK;
        break;
    case E2E_P04STATUS_OKSOMELOST:
        return E2E_P_OK;
        break;
    case E2E_P04STATUS_ERROR:
        return E2E_P_ERROR;
        break;
    case E2E_P04STATUS_REPEATED:
        return E2E_P_REPEATED;
        break;
    case E2E_P04STATUS_NONEWDATA:
        return E2E_P_NONEWDATA;
        break;
    case E2E_P04STATUS_WRONGSEQUENCE:
        return E2E_P_WRONGSEQUENCE;
        break;
    default:
        break;
    }
    return E2E_P_RESERVED;        
}

Std_ReturnType E2E_P04ProtectInit(E2E_P04ProtectStateType *StatePtr) {
    if(StatePtr == NULL)
        return E2E_E_INPUTERR_NULL;
    StatePtr->Counter = 0;
    return E2E_E_OK;
}

Std_ReturnType E2E_P04CheckInit(E2E_P04CheckStateType *StatePtr) {
    if(StatePtr == NULL)
        return E2E_E_INPUTERR_NULL;
    StatePtr->Counter = 0xFFFF;
    StatePtr->Status  = E2E_P04STATUS_ERROR;
    return E2E_E_OK;
}