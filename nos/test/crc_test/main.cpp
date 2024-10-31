#include <iostream>

#include "crc/include/crc.h"

int main() {
    uint8_t data[10][10] = {{0x00, 0x00, 0x00, 0x00}, {0xf2, 0x01, 0x83},      {0x0f, 0xaa, 0x00, 0x55}, {0x00, 0xff, 0x55, 0x11}, {0x33, 0x22, 0x55, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
                            {0x92, 0x6B, 0x55},       {0xff, 0xff, 0xff, 0xff}};
    uint32_t len[10] = {4, 3, 4, 4, 9, 3, 4};

    printf("-------------Crc_CalculateCRC8:-----------------\n");
    for (uint8_t i = 0; i < 7; i++) {
        uint8_t ComputedCRC = Crc_CalculateCRC8(data[i], len[i], 0, true);
        printf("ComputedCRC:%02X\n", ComputedCRC);
    }
    printf("-------------Crc_CalculateCRC8H2F:--------------\n");
    for (uint8_t i = 0; i < 7; i++) {
        uint8_t ComputedCRC = Crc_CalculateCRC8H2F(data[i], len[i], 0, true);
        printf("ComputedCRC:%02X\n", ComputedCRC);
    }
    printf("-------------Crc_CalculateCRC16:----------------\n");
    for (uint8_t i = 0; i < 7; i++) {
        uint16_t ComputedCRC = Crc_CalculateCRC16(data[i], len[i], 0, true);
        printf("ComputedCRC:%04X\n", ComputedCRC);
    }
    printf("-------------Crc_CalculateCRC16ARC:-------------\n");
    for (uint8_t i = 0; i < 7; i++) {
        uint16_t ComputedCRC = Crc_CalculateCRC16ARC(data[i], len[i], 0, true);
        printf("ComputedCRC:%04X\n", ComputedCRC);
    }
    printf("-------------Crc_CalculateCRC32:----------------\n");
    for (uint8_t i = 0; i < 7; i++) {
        uint32_t ComputedCRC = Crc_CalculateCRC32(data[i], len[i], 0, true);
        printf("ComputedCRC:%08X\n", ComputedCRC);
    }
    printf("-------------Crc_CalculateCRC32P4:--------------\n");
    for (uint8_t i = 0; i < 7; i++) {
        uint32_t ComputedCRC = Crc_CalculateCRC32P4(data[i], len[i], 0, true);
        printf("ComputedCRC:%08X\n", ComputedCRC);
    }
    printf("-------------Crc_CalculateCRC64:----------------\n");
    for (uint8_t i = 0; i < 7; i++) {
        uint64_t ComputedCRC = Crc_CalculateCRC64(data[i], len[i], 0, true);
        printf("ComputedCRC:%0llX\n", (long long unsigned int)ComputedCRC);
    }

    return 0;
}
