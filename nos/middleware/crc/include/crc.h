/*
* Copyright (c) Hozon Auto Co., Ltd. 2022-2023. All rights reserved.
* Description: CRCLibrary
*/

#ifndef CRC_H
#define CRC_H

#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif

uint8_t Crc_CalculateCRC8(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                uint8_t Crc_StartValue8, uint8_t Crc_IsFirstCall);

uint8_t Crc_CalculateCRC8H2F(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                   uint8_t Crc_StartValue8H2F, uint8_t Crc_IsFirstCall);

uint16_t Crc_CalculateCRC16(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                  uint16_t Crc_StartValue16, uint8_t Crc_IsFirstCall);

uint16_t Crc_CalculateCRC16ARC(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                     uint16_t Crc_StartValue16ARC, uint8_t Crc_IsFirstCall);

uint32_t Crc_CalculateCRC32(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                  uint32_t Crc_StartValue32, uint8_t Crc_IsFirstCall);

uint32_t Crc_CalculateCRC32P4(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                  uint32_t Crc_StartValue32P4, uint8_t Crc_IsFirstCall);

uint64_t Crc_CalculateCRC64(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                  uint64_t Crc_StartValue64, uint8_t Crc_IsFirstCall);

#ifdef __cplusplus
}
#endif
#endif

