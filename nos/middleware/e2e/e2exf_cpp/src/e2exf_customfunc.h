#ifndef E2EXF_CUSTOM_H
#define E2EXF_CUSTOM_H

void Profile22_custom_before(uint8_t* buffer, uint32_t* bufferLength) {
    int32_t i;
    uint8_t crc = buffer[7], counter = buffer[6];
    if (*bufferLength > 8) {
        for (i = *bufferLength - 1; i >= 8; i--) buffer[i + 1] = buffer[i];
        for (i = 6; i >= 0; i--) buffer[i + 2] = buffer[i];
        if(!(counter & 0xF0))
        {
            buffer[0] = 0u;
            buffer[1] = 0u;
            buffer[8] = (counter + 1) % 16;
        } else {
            buffer[0] = crc;
            buffer[1] = counter & 0x0F;
            buffer[8] = counter & 0x0F;
        }
    } else {
        for (i = *bufferLength - 2; i >= 0; i--) buffer[i + 2] = buffer[i];
        if(!(counter & 0xF0))
        {
            buffer[0] = 0u;
            buffer[1] = 0u;
            buffer[*bufferLength] = (counter + 1) % 16;
        } else {
            buffer[0] = crc;
            buffer[1] = counter & 0x0F;
            buffer[*bufferLength] = counter & 0x0F;
        }
    }
    *bufferLength += 1;
}

void Profile22_custom_after(uint8_t* buffer, uint32_t* bufferLength) {
    int32_t i;
    uint8_t crc = buffer[0], counter = buffer[1] & 0x0Fu;
    if (*bufferLength > 8) {
        for (i = 2; i <= 7; i++) buffer[i - 2] = buffer[i];
        for (i = 8; i < (int32_t)*bufferLength; i++) buffer[i - 1] = buffer[i];
        buffer[6] = counter;
        buffer[7] = crc;
    } else {
        for (i = 2; i < (int32_t)*bufferLength - 2; i++) buffer[i - 2] = buffer[i];
        buffer[*bufferLength - 2] = counter;
        buffer[*bufferLength - 1] = crc;
    }
    *bufferLength -= 1;
}

#endif