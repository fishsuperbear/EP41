/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: The header file of Calculating CRC including all CRC algorithms' definition
 * Create: 2019-06-17
 */

#ifndef CRC_H
#define CRC_H

#ifdef __cplusplus
extern "C"
{
#endif

unsigned char Crc_CalculateCRC8(const unsigned char *Crc_DataPtr, unsigned int Crc_Length,
                                unsigned char Crc_StartValue8, unsigned char Crc_IsFirstCall);

unsigned char Crc_CalculateCRC8H2F(const unsigned char *Crc_DataPtr, unsigned int Crc_Length,
                                   unsigned char Crc_StartValue8, unsigned char Crc_IsFirstCall);

unsigned short Crc_CalculateCRC16(const unsigned char *Crc_DataPtr, unsigned int Crc_Length,
                                  unsigned short Crc_StartValue16, unsigned char Crc_IsFirstCall);

unsigned short Crc_CalculateCRC16ARC(const unsigned char *Crc_DataPtr, unsigned int Crc_Length,
                                     unsigned short Crc_StartValue16, unsigned char Crc_IsFirstCall);

unsigned int Crc_CalculateCRC32(const unsigned char *Crc_DataPtr, unsigned int Crc_Length,
                                unsigned int Crc_StartValue32, unsigned char Crc_IsFirstCall);

unsigned int Crc_CalculateCRC32P4(const unsigned char *Crc_DataPtr, unsigned int Crc_Length,
                                  unsigned int Crc_StartValue32, unsigned char Crc_IsFirstCall);

unsigned long long Crc_CalculateCRC64(const unsigned char *Crc_DataPtr, unsigned int Crc_Length,
                                      unsigned long long Crc_StartValue64, unsigned char Crc_IsFirstCall);

#ifdef __cplusplus
}
#endif
#endif

