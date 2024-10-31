#include "e2e/e2exf/include/e2exf.h"

boolean E2EXf_Initialized = FALSE;

void E2EXf_P04_ProtectInit(E2E_P04ProtectStateType* stateptr) {
    E2E_P04ProtectInit(stateptr);  //
}

void E2EXf_P04_CheckInit(E2E_P04CheckStateType* stateptr) {
    E2E_P04CheckInit(stateptr);  //
}

void E2EXf_P22_ProtectInit(E2E_P22ProtectStateType* stateptr) {
    E2E_P22ProtectInit(stateptr);  //
}

void E2EXf_P22_CheckInit(E2E_P22CheckStateType* stateptr) {
    E2E_P22CheckInit(stateptr);  //
}

boolean E2EXf_input_checks(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config) {
    if (config->InPlace == TRUE) {
        // [SWS_E2EXf_00102] 1.
        if ((buffer == NULL) || (buffer != NULL && inputBufferLength < config->upperHeaderBitsToShift / 8u)) return FALSE;
        // [SWS_E2EXf_00102] 2.
        if (bufferLength == NULL) return FALSE;
    } else {
        // [SWS_E2EXf_00106] 1.
        if ((inputBuffer == NULL) || (inputBuffer != NULL && inputBufferLength < config->upperHeaderBitsToShift / 8u)) return FALSE;
        // [SWS_E2EXf_00106] 2.
        if (bufferLength == NULL) return FALSE;
        // [SWS_E2EXf_00106] 3.
        if (buffer == NULL) return FALSE;
    }
    return TRUE;
}

boolean E2EXf_Inv_input_checks(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config) {
    if (config->InPlace == TRUE) {
        // [SWS_E2EXf_00105] 1.
        if ((buffer == NULL && inputBufferLength != 0) || (buffer != NULL && inputBufferLength < config->headerLength / 8u + config->upperHeaderBitsToShift / 8u)) return FALSE;
        // [SWS_E2EXf_00105] 2.
        if (bufferLength == NULL) return FALSE;
    } else {
        // [SWS_E2EXf_00103] 1.
        if ((inputBuffer == NULL && inputBufferLength != 0) || (inputBuffer != NULL && inputBufferLength < config->headerLength / 8u + config->upperHeaderBitsToShift / 8u)) return FALSE;
        // [SWS_E2EXf_00103] 2.
        if (bufferLength == NULL) return FALSE;
        // [SWS_E2EXf_00103] 3.
        if (buffer == NULL) return FALSE;
    }
    return TRUE;
}

boolean E2EXf_handling_P01_P02(uint32_t* bufferLength) {
    //! P01 P02 is not realize yet
    return TRUE;
}

void E2EXf_MoveHeader(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config) {
    uint32_t HeaderBytesToShift = config->upperHeaderBitsToShift >> 3u;
    uint32_t HeaderBytesLength = config->headerLength >> 3u;
    if (HeaderBytesToShift > 0) {
        if (config->InPlace == TRUE) {
            // [SWS_E2EXf_00108]
            for (uint32_t i = 0; i < HeaderBytesToShift; ++i) {
                buffer[i] = buffer[i + HeaderBytesLength];
            }
        } else {
            // [SWS_E2EXf_00109]
            for (uint32_t i = 0; i < HeaderBytesToShift; ++i) {
                buffer[i] = inputBuffer[i];
            }
            for (uint32_t i = HeaderBytesToShift; i < inputBufferLength; ++i) {
                buffer[i + HeaderBytesLength] = inputBuffer[i];
            }
        }
        // [SWS_E2EXf_00115]
    } else if (HeaderBytesToShift == 0 && config->InPlace == FALSE) {
        for (uint32_t i = 0; i < inputBufferLength; ++i) {
            buffer[i + HeaderBytesLength] = inputBuffer[i];
        }
    }
}

/*DRAFT*/ void E2EXf_Inv_MoveHeader(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config) {
    uint32_t HeaderBytesToShift = config->upperHeaderBitsToShift >> 3u;
    uint32_t HeaderBytesLength = config->headerLength >> 3u;
    // [SWS_E2EXf_00112]
    if (buffer != NULL && config->upperHeaderBitsToShift > 0 && config->InPlace == TRUE) {
        for (uint32_t i = HeaderBytesToShift - 1;;) {
            buffer[i + HeaderBytesLength] = buffer[i];
            if (i == 0) break;
            i--;
        }
    }
    // [SWS_E2EXf_00113]
    else if (inputBuffer != NULL && config->upperHeaderBitsToShift > 0 && config->InPlace == FALSE) {
        for (uint32_t i = 0; i < HeaderBytesToShift; i++) {
            buffer[i + HeaderBytesLength] = inputBuffer[i];
        }
        for (uint32_t i = HeaderBytesLength + HeaderBytesToShift; i < inputBufferLength; i++) {
            buffer[i - HeaderBytesToShift] = inputBuffer[i];
        }
    }
    // [SWS_E2EXf_00116]
    else if (inputBuffer != NULL && config->upperHeaderBitsToShift == 0 && config->InPlace == FALSE) {
        for (uint32_t i = 0; i < inputBufferLength; ++i) {
            buffer[i] = inputBuffer[i];
        }
    }
}

E2E_PCheckStatusType E2EXf_MapCodeToStatus(Std_TransformerForwardCode forwardedCode) {
    // [SWS_E2EXf_00208]
    switch (forwardedCode) {
        case E_OK:
            return E2E_P_OK;
        case E_SAFETY_INVALID_REP:
            return E2E_P_REPEATED;
        case E_SAFETY_INVALID_SEQ:
            return E2E_P_WRONGSEQUENCE;
        // [DRAFT] E_SAFETY_INVALID_CRC == E_SAFETY_INVALID_ERR
        case E_SAFETY_INVALID_ERR:
            return E2E_P_ERROR;
        default:
            return E2E_P_OK;
    }
    return E2E_P_OK;
}

/*------------------------------------Profile 04------------------------------------------*/

uint8_t E2EXf_P04(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P04ProtectStateType* state,
                  Std_TransformerForwardCode forwardedCode) {
    uint8_t ret = E_SAFETY_HARD_RUNTIMEERROR;
    Std_ReturnType prot_ret;
    // 1.E2EXf_input_checks
    if (FALSE == E2EXf_input_checks(buffer, bufferLength, inputBuffer, inputBufferLength, config)) return ret;
    // [SWS_E2EXf_00111]
    *bufferLength = inputBufferLength + config->headerLength / 8u;
    // 2.E2EXf_handling_P01_P02
    if (FALSE == E2EXf_handling_P01_P02(bufferLength)) return ret;
    // 3.Copy buffer / Header according to upperHeaderBitsToShift and headerLength
    E2EXf_MoveHeader(buffer, bufferLength, inputBuffer, inputBufferLength, config);
    // [SWS_E2EXf_00168]
    if (config->DataTransformationStatusForwarding == transformerStatusForwarding) {
        prot_ret = E2E_P04Forward(&config->ProfileConfig.Profile04, state, buffer, *bufferLength, E2EXf_MapCodeToStatus(forwardedCode));
    } else  // [SWS_E2EXf_00107]
        prot_ret = E2E_P04Protect(&config->ProfileConfig.Profile04, state, buffer, *bufferLength);
    // [SWS_E2EXf_00018]
    if (prot_ret == E2E_E_OK) ret = E2E_E_OK;
    return ret;
}

uint8_t E2EXf_Inv_P04(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P04CheckStateType* state,
                      const E2E_SMConfigType* smConfig, E2E_SMCheckStateType* smState) {
    uint8_t ret = E_SAFETY_HARD_RUNTIMEERROR;
    Std_ReturnType checkret;
    E2E_PCheckStatusType checkStatus;
    // 1.E2EXf_input_checks
    if (FALSE == E2EXf_Inv_input_checks(buffer, bufferLength, inputBuffer, inputBufferLength, config)) return ret;
    // [SWS_E2EXf_00154]
    if (TRUE == config->disableEndToEndCheck) {
        ret = E2E_E_OK;
        for (uint32_t i = 0; i < inputBufferLength; ++i) buffer[i] = inputBuffer[i];
        return ret;
    }
    // 2.E2EXf_Inv_handling_P01_P02

    // [SWS_E2EXf_00104], [SWS_E2EXf_00124]
    if (config->InPlace == TRUE)  // in-place
        checkret = E2E_P04Check(&config->ProfileConfig.Profile04, state, buffer, inputBufferLength);
    else  // out-of-place
        checkret = E2E_P04Check(&config->ProfileConfig.Profile04, state, inputBuffer, inputBufferLength);

    // 3.E2EXf_Inv_handling_P01_P02_forceConstantMaxDeltaCounter

    // 4.E2EXf_Inv_handle_Statemachine
    checkStatus = E2E_P04MapStatusToSM(checkret, state->Status);
    // if (PortInterface == ClientServerInterface && PortPrototype == RPortPrototype && checkStatus != E2E_P_OK)
    //     checkStatus = E2E_P_ERROR;

    // [SWS_E2EXf_00027]
    if (TRUE == config->disableEndToEndStatemachine) {
        ret = 0x60 | (checkStatus & 0x0F);
    } else {
        ret = E2E_SMCheck(checkStatus, smConfig, smState);
    }
    // [SWS_E2EXf_00112], [SWS_E2EXf_00113], [SWS_E2EXf_00116]
    E2EXf_Inv_MoveHeader(buffer, bufferLength, inputBuffer, inputBufferLength, config);
    // [SWS_E2EXf_00114]
    if (inputBufferLength == 0)
        *bufferLength = 0;
    else
        *bufferLength = inputBufferLength - config->headerLength / 8;
    return ret;
}

/*------------------------------------Profile 22------------------------------------------*/

uint8_t E2EXf_P22(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22ProtectStateType* state,
                  Std_TransformerForwardCode forwardedCode) {
    uint8_t ret = E_SAFETY_HARD_RUNTIMEERROR;
    Std_ReturnType prot_ret;
    // 1.E2EXf_input_checks
    if (FALSE == E2EXf_input_checks(buffer, bufferLength, inputBuffer, inputBufferLength, config)) return ret;
    // [SWS_E2EXf_00111]
    *bufferLength = inputBufferLength + config->headerLength / 8u;
    // 2.E2EXf_handling_P01_P02
    if (FALSE == E2EXf_handling_P01_P02(bufferLength)) return ret;
    // 3.Copy buffer / Header according to upperHeaderBitsToShift and headerLength
    E2EXf_MoveHeader(buffer, bufferLength, inputBuffer, inputBufferLength, config);
    // [SWS_E2EXf_00168]
    if (config->DataTransformationStatusForwarding == transformerStatusForwarding) {
        prot_ret = E2E_P22Forward(&config->ProfileConfig.Profile22, state, buffer, *bufferLength, E2EXf_MapCodeToStatus(forwardedCode));
    } else  // [SWS_E2EXf_00107]
        prot_ret = E2E_P22Protect(&config->ProfileConfig.Profile22, state, buffer, *bufferLength);
    // [SWS_E2EXf_00018]
    if (prot_ret == E2E_E_OK) ret = E2E_E_OK;
    return ret;
}

uint8_t E2EXf_Inv_P22(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22CheckStateType* state,
                      const E2E_SMConfigType* smConfig, E2E_SMCheckStateType* smState) {
    uint8_t ret = E_SAFETY_HARD_RUNTIMEERROR;
    Std_ReturnType checkret;
    E2E_PCheckStatusType checkStatus;
    // 1.E2EXf_input_checks
    if (FALSE == E2EXf_Inv_input_checks(buffer, bufferLength, inputBuffer, inputBufferLength, config)) return ret;
    // [SWS_E2EXf_00154]
    if (TRUE == config->disableEndToEndCheck) {
        ret = E2E_E_OK;
        for (uint32_t i = 0; i < inputBufferLength; ++i) buffer[i] = inputBuffer[i];
        return ret;
    }
    // 2.E2EXf_Inv_handling_P01_P02

    // [SWS_E2EXf_00122], [SWS_E2EXf_00124]
    if (config->InPlace == TRUE)  // in-place
        checkret = E2E_P22Check(&config->ProfileConfig.Profile22, state, buffer, inputBufferLength);
    else  // out-of-place
        checkret = E2E_P22Check(&config->ProfileConfig.Profile22, state, inputBuffer, inputBufferLength);

    // 3.E2EXf_Inv_handling_P01_P02_forceConstantMaxDeltaCounter

    // 4.E2EXf_Inv_handle_Statemachine
    checkStatus = E2E_P22MapStatusToSM(checkret, state->Status);
    // if (PortInterface == ClientServerInterface && PortPrototype == RPortPrototype && checkStatus != E2E_P_OK)
    //     checkStatus = E2E_P_ERROR;

    // [SWS_E2EXf_00027]
    if (TRUE == config->disableEndToEndStatemachine) {
        ret = 0x60 | (checkStatus & 0x0F);
    } else {
        ret = E2E_SMCheck(checkStatus, smConfig, smState);
    }
    // [SWS_E2EXf_00112], [SWS_E2EXf_00113], [SWS_E2EXf_00116]
    E2EXf_Inv_MoveHeader(buffer, bufferLength, inputBuffer, inputBufferLength, config);
    // [SWS_E2EXf_00114]
    if (inputBufferLength == 0)
        *bufferLength = 0;
    else
        *bufferLength = inputBufferLength - config->headerLength / 8;
    return ret;
}

/*------------------------------------Profile custom------------------------------------------*/
uint8_t E2EXf_P22_CUSTOM(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22ProtectStateType* state,
                         Std_TransformerForwardCode forwardedCode, void (*custom_before)(uint8_t*, uint32_t*), void (*custom_after)(uint8_t*, uint32_t*)) {
    uint8_t ret = E_SAFETY_HARD_RUNTIMEERROR;
    Std_ReturnType prot_ret;

    // 1.E2EXf_input_checks
    if (FALSE == E2EXf_input_checks(buffer, bufferLength, inputBuffer, inputBufferLength, config)) return ret;
    // [SWS_E2EXf_00111]
    *bufferLength = inputBufferLength + config->headerLength / 8u;
    // 2.E2EXf_handling_P01_P02
    if (FALSE == E2EXf_handling_P01_P02(bufferLength)) return ret;
    // 3.Copy buffer / Header according to upperHeaderBitsToShift and headerLength
    E2EXf_MoveHeader(buffer, bufferLength, inputBuffer, inputBufferLength, config);
    // [SWS_E2EXf_00168]

    if (custom_before != NULL) {
        buffer[6] = state->Counter & 0x0F;
        custom_before(buffer, bufferLength);
    }

    if (config->DataTransformationStatusForwarding == transformerStatusForwarding) {
        prot_ret = E2E_P22Forward(&config->ProfileConfig.Profile22, state, buffer, *bufferLength, E2EXf_MapCodeToStatus(forwardedCode));
    } else  // [SWS_E2EXf_00107]
        prot_ret = E2E_P22Protect(&config->ProfileConfig.Profile22, state, buffer, *bufferLength);

    if (custom_after != NULL) {
        custom_after(buffer, bufferLength);
    }

    // [SWS_E2EXf_00018]
    if (prot_ret == E2E_E_OK) ret = E2E_E_OK;
    return ret;
}

uint8_t E2EXf_Inv_P22_CUSTOM(uint8_t* buffer, uint32_t* bufferLength, const uint8_t* inputBuffer, uint32_t inputBufferLength, const E2EXf_ConfigType* config, E2E_P22CheckStateType* state,
                             const E2E_SMConfigType* smConfig, E2E_SMCheckStateType* smState, void (*custom_before)(uint8_t*, uint32_t*), void (*custom_after)(uint8_t*, uint32_t*)) {
    uint8_t ret = E_SAFETY_HARD_RUNTIMEERROR;
    Std_ReturnType checkret;
    E2E_PCheckStatusType checkStatus;
    // 1.E2EXf_input_checks
    if (FALSE == E2EXf_Inv_input_checks(buffer, bufferLength, inputBuffer, inputBufferLength, config)) return ret;
    // [SWS_E2EXf_00154]
    if (TRUE == config->disableEndToEndCheck) {
        ret = E2E_E_OK;
        for (uint32_t i = 0; i < inputBufferLength; ++i) buffer[i] = inputBuffer[i];
        return ret;
    }
    // 2.E2EXf_Inv_handling_P01_P02

    if (custom_before != NULL) {
        buffer[6] = buffer[6] | 0xF0;
        custom_before(buffer, &inputBufferLength);
    }

    // [SWS_E2EXf_00122], [SWS_E2EXf_00124]
    if (config->InPlace == TRUE)  // in-place
        checkret = E2E_P22Check(&config->ProfileConfig.Profile22, state, buffer, inputBufferLength);
    else  // out-of-place
        checkret = E2E_P22Check(&config->ProfileConfig.Profile22, state, inputBuffer, inputBufferLength);

    if (custom_after != NULL) {
        custom_after(buffer, &inputBufferLength);
        
    }

    // 3.E2EXf_Inv_handling_P01_P02_forceConstantMaxDeltaCounter

    // 4.E2EXf_Inv_handle_Statemachine
    checkStatus = E2E_P22MapStatusToSM(checkret, state->Status);
    // if (PortInterface == ClientServerInterface && PortPrototype == RPortPrototype && checkStatus != E2E_P_OK)
    //     checkStatus = E2E_P_ERROR;

    // [SWS_E2EXf_00027]
    if (TRUE == config->disableEndToEndStatemachine) {
        ret = 0x60 | (checkStatus & 0x0F);
    } else {
        ret = E2E_SMCheck(checkStatus, smConfig, smState);
    }

    // [SWS_E2EXf_00112], [SWS_E2EXf_00113], [SWS_E2EXf_00116]
    E2EXf_Inv_MoveHeader(buffer, bufferLength, inputBuffer, inputBufferLength, config);
    // [SWS_E2EXf_00114]
    if (inputBufferLength == 0)
        *bufferLength = 0;
    else
        *bufferLength = inputBufferLength - config->headerLength / 8;

    return ret;
}
