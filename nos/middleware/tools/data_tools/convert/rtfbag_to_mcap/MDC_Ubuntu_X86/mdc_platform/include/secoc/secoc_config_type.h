/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: The declaration of the SecOCConfigType of SecOC
 * Create: 2020-10-09
 */

#ifndef SECOC_CONFIG_TYPE_H
#define SECOC_CONFIG_TYPE_H

#include <stdint.h>
#include <stdbool.h>
#include "secoc/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t dataId;
    bool enableFreshnessValue;
    uint16_t freshnessValueId;
    uint16_t freshnessValueLength; // 0-64 bits
    OptionalValue truncatedFVInfo;
    OptionalValue truncatedAuthInfo;
    SecOCAlgoConfig algoConfig;
    SecOCDefaultPattern defaultPattern;
    SecOCAttempts attempts;
    SecOCByteAlignmentType maxAlignScalarType; // Byte Alignment Requirements
    bool ignoreVerificationResult;
    SecOCVerificationStatusPropagationMode propagationMode;
    SecOCSecuredArea area;
    SecOCUseAuthDataFreshness useAuthDataFreshness;
    bool useTxConfirmation;
} SecOCConfigType;

#ifdef __cplusplus
}
#endif

#endif // SECOC_CONFIG_TYPE_H
