#include "e2e/e2e/include/e2e_sm.h"

Std_ReturnType E2E_SMCheckInit(E2E_SMCheckStateType* StatePtr, 
                               const E2E_SMConfigType* ConfigPtr) {
    if(StatePtr == NULL || ConfigPtr == NULL)
        return E2E_E_INPUTERR_NULL;
    StatePtr->ErrorCount = 0;
    StatePtr->OkCount = 0;
    StatePtr->WindowTopIndex = 0;
    StatePtr->SMState = E2E_SM_NODATA;
    // uint8_t *arr=(uint8_t*)malloc(ConfigPtr->WindowSizeInit*sizeof(uint8_t));
    for(uint8_t i = 0; i < ConfigPtr->WindowSizeInit; ++i) {
        *(StatePtr->ProfileStatusWindow + i) = E2E_P_NOTAVAILABLE;
    }
    // StatePtr->ProfileStatusWindow = arr;
    return E2E_E_OK;
}

Std_ReturnType E2E_SMCheck(E2E_PCheckStatusType ProfileStatus, const E2E_SMConfigType* ConfigPtr,
                           E2E_SMCheckStateType* StatePtr) {
    if(E2E_SM_DEINIT == StatePtr->SMState && 
       E2E_E_INPUTERR_NULL == E2E_SMCheckInit(StatePtr,ConfigPtr))
            return E2E_E_WRONGSTATE;

    E2E_SMAddStatus(ProfileStatus,ConfigPtr,StatePtr);

    switch (StatePtr->SMState)
    {
case E2E_SM_NODATA:
        if(StatePtr->WindowTopIndex == 0) {
            if(ConfigPtr->ClearToInvalid == TRUE) 
                E2E_SMClearStatus(StatePtr, ConfigPtr);
            else 
                E2E_SMClearRemainingStatus(StatePtr, ConfigPtr, E2E_SM_INVALID);
            StatePtr->SMState = E2E_SM_INVALID;
        }
        else if(ProfileStatus != E2E_P_ERROR && ProfileStatus != E2E_P_NONEWDATA) {
            E2E_SMClearStatus(StatePtr, ConfigPtr);
            StatePtr->SMState = E2E_SM_INIT;
        }
break;

case E2E_SM_INIT:
        if((StatePtr->ErrorCount <= ConfigPtr->MaxErrorStateInit) && 
           (StatePtr->OkCount    >= ConfigPtr->MinOkStateInit   )) {
            E2E_SMClearRemainingStatus(StatePtr, ConfigPtr, E2E_SM_VALID);
            StatePtr->SMState = E2E_SM_VALID;
        }
        else if((StatePtr->ErrorCount > ConfigPtr->MaxErrorStateInit) || 
              (((StatePtr->WindowTopIndex - 1) % ConfigPtr->WindowSizeInit) + 1 - StatePtr->OkCount > 
                 ConfigPtr->WindowSizeInit - ConfigPtr->MinOkStateInit)) {
            if(ConfigPtr->ClearToInvalid == TRUE) 
                E2E_SMClearStatus(StatePtr, ConfigPtr);
            else 
                E2E_SMClearRemainingStatus(StatePtr, ConfigPtr, E2E_SM_INVALID);
            StatePtr->SMState = E2E_SM_INVALID;
        }
break;

case E2E_SM_INVALID:
        if((StatePtr->ErrorCount <= ConfigPtr->MaxErrorStateInvalid) && 
           (StatePtr->OkCount    >= ConfigPtr->MinOkStateInvalid   )) {
            E2E_SMClearRemainingStatus(StatePtr, ConfigPtr, E2E_SM_VALID);
            StatePtr->SMState = E2E_SM_VALID;
        }
        else if(StatePtr->ErrorCount > ConfigPtr->MaxErrorStateInvalid) {
        //Document ID 849: AUTOSAR_PRS_E2EProtocol P279 with some problem
            E2E_SMClearStatus(StatePtr, ConfigPtr);
        }
break;

case E2E_SM_VALID:
        if(!((StatePtr->ErrorCount <= ConfigPtr->MaxErrorStateInvalid) && 
             (StatePtr->OkCount    >= ConfigPtr->MinOkStateInvalid   ))){
            if(ConfigPtr->ClearToInvalid == TRUE) 
                E2E_SMClearStatus(StatePtr, ConfigPtr);
            StatePtr->SMState = E2E_SM_INVALID;
        }
break;

    default: break;
    }
    return E2E_E_OK;
}

inline uint8_t E2E_ChooseWindowSize(E2E_SMStateType CurrentState, const E2E_SMConfigType* ConfigPtr) {
    return  CurrentState == E2E_SM_INIT  ? ConfigPtr->WindowSizeInit    : 
           (CurrentState == E2E_SM_VALID ? ConfigPtr->WindowSizeValid   : 
                                           ConfigPtr->WindowSizeInvalid);
}

inline void E2E_SMAddStatus(E2E_PCheckStatusType ProfileStatus, const E2E_SMConfigType* ConfigPtr,
                            E2E_SMCheckStateType* StatePtr) {                         
    *(StatePtr->ProfileStatusWindow + StatePtr->WindowTopIndex) = ProfileStatus;

    uint8_t OkCount = 0,    ErrorCount = 0,
            CurrentWindowSize = E2E_ChooseWindowSize(StatePtr->SMState, ConfigPtr);
    for(uint8_t i = 0; i < CurrentWindowSize; i++) {
        if(*(StatePtr->ProfileStatusWindow + i) == E2E_P_OK)
            OkCount++;
        else if(*(StatePtr->ProfileStatusWindow + i) == E2E_P_ERROR)
            ErrorCount++;
    }
    StatePtr->OkCount = OkCount;
    StatePtr->ErrorCount = ErrorCount;
    if(StatePtr->WindowTopIndex == CurrentWindowSize - 1)
        StatePtr->WindowTopIndex = 0;
    else
        StatePtr->WindowTopIndex++;
}

inline void E2E_SMClearStatus(E2E_SMCheckStateType* StatePtr, const E2E_SMConfigType* ConfigPtr) {

    StatePtr->ErrorCount = 0;
    StatePtr->OkCount = 0;
    StatePtr->WindowTopIndex = 0;
    for(uint8_t i = 0; i < ConfigPtr->WindowSizeValid; ++i) {
        *(StatePtr->ProfileStatusWindow + i) = E2E_P_NOTAVAILABLE;
    }
}

inline void E2E_SMClearRemainingStatus(E2E_SMCheckStateType* StatePtr, 
                                       const E2E_SMConfigType* ConfigPtr, E2E_SMStateType NextState) {
    uint8_t CurrentWindowSize = E2E_ChooseWindowSize(StatePtr->SMState, ConfigPtr),
            NextWindowSize    = E2E_ChooseWindowSize(NextState        , ConfigPtr);
    uint8_t ClearCount        = CurrentWindowSize,index = StatePtr->WindowTopIndex;

    if(CurrentWindowSize < NextWindowSize) {
        while(ClearCount --> 0) {
            *(StatePtr->ProfileStatusWindow + index) = E2E_P_NONEWDATA;
            index = (index ? index : CurrentWindowSize) - 1;
        }
    }
}