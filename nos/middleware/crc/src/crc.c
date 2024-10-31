#include "crc/include/crc.h"
#include "crc/src/crc_tables.h"

/*!  
 *  Specification        : [SWS_Crc_00030]
 *  CRC result width     : 8 bits
 *  Polynomial           : 1Dh
 *  Initial value        : FFh
 *  Input data reflected : No
 *  Result data reflected: No
 *  XOR value            : FFh          
 */
uint8_t Crc_CalculateCRC8(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                uint8_t Crc_StartValue8, uint8_t Crc_IsFirstCall) {
    uint8_t Crc = Crc_IsFirstCall ? 0xFFu : ~Crc_StartValue8;
    for(uint32_t byteindex = 0; byteindex < Crc_Length ; ++byteindex ) {
        Crc ^= *(Crc_DataPtr + byteindex);
        #ifdef CRC_TABLE_DISABLE
        for(uint8_t bit = 0; bit < 8u; ++bit){
            if((Crc & 0x80u) != 0)
                Crc = (Crc << 1u) ^ 0x1Du;
            else
                Crc = (Crc << 1u);
        }
        #else
            Crc = (Crc << 8u) ^ CRC8_TABLE[Crc]; 
        #endif
    }
    return ~Crc;
}


/*!  
 *  Specification        : [SWS_Crc_00043]
 *  CRC result width     : 8 bits
 *  Polynomial           : 2Fh
 *  Initial value        : FFh
 *  Input data reflected : No
 *  Result data reflected: No
 *  XOR value            : FFh          
 */
uint8_t Crc_CalculateCRC8H2F(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                   uint8_t Crc_StartValue8H2F, uint8_t Crc_IsFirstCall) {
    uint8_t Crc = Crc_IsFirstCall ? 0xFFu : ~Crc_StartValue8H2F;
    for(uint32_t byteindex = 0; byteindex < Crc_Length ; ++byteindex ) {
        Crc ^= *(Crc_DataPtr + byteindex);
        #ifdef CRC_TABLE_DISABLE
        for(uint8_t bit = 0; bit < 8u; ++bit){
            if((Crc & 0x80u) != 0)
                Crc = (Crc << 1u) ^ 0x2Fu;
            else
                Crc = (Crc << 1u);
        }
        #else
            Crc = (Crc << 8u) ^ CRC8H2F_TABLE[Crc]; 
        #endif
    }
    return ~Crc;
}


/*!  
 *  Specification        : [SWS_Crc_00002]
 *  CRC result width     : 16 bits
 *  Polynomial           : 1021h
 *  Initial value        : FFFFh
 *  Input data reflected : No
 *  Result data reflected: No
 *  XOR value            : 0000h         
 */
uint16_t Crc_CalculateCRC16(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                  uint16_t Crc_StartValue16, uint8_t Crc_IsFirstCall) {
    uint16_t Crc = Crc_IsFirstCall ? 0xFFFFu : Crc_StartValue16;
    for(uint32_t byteindex = 0; byteindex < Crc_Length ; ++byteindex ) {
        Crc ^= (*(Crc_DataPtr + byteindex) << 8u);
        #ifdef CRC_TABLE_DISABLE
        for(uint8_t bit = 0; bit < 8u; ++bit){
            if((Crc & 0x8000u) != 0)
                Crc = (Crc << 1u) ^ 0x1021u;
            else
                Crc = (Crc << 1u);
        }
        #else
            Crc = (Crc << 8u) ^ CRC16_TABLE[(uint8_t)(Crc>>8)]; 
        #endif
    }
    return Crc;
}


/*!  
 *  Specification        : [SWS_Crc_00067]
 *  CRC result width     : 16 bits
 *  Polynomial           : 8005h
 *  Initial value        : 0000h
 *  Input data reflected : Yes
 *  Result data reflected: Yes
 *  XOR value            : 0000h           
 */
uint16_t Crc_CalculateCRC16ARC(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                     uint16_t Crc_StartValue16ARC, uint8_t Crc_IsFirstCall) {
    uint16_t Crc = Crc_IsFirstCall ? 0x0000u : Crc_StartValue16ARC;
    for(uint32_t byteindex = 0; byteindex < Crc_Length ; ++byteindex ) {
        Crc ^= *(Crc_DataPtr + byteindex);
        #ifdef CRC_TABLE_DISABLE
        for(uint8_t bit = 0; bit < 8u; ++bit){
            if((Crc & 0x1u) != 0)
                Crc = (Crc >> 1u) ^ 0xA001u; //reflected 0x8005
            else
                Crc = (Crc >> 1u);
        }
        #else
            Crc = (Crc >> 8u) ^ CRC16ARC_TABLE[(uint8_t)Crc]; 
        #endif
    }
    return Crc;
}


/*!  
 *  Specification        : [SWS_Crc_00003]
 *  CRC result width     : 32 bits
 *  Polynomial           : 04C11DB7h
 *  Initial value        : FFFFFFFFh
 *  Input data reflected : Yes
 *  Result data reflected: Yes
 *  XOR value            : FFFFFFFFh        
 */
uint32_t Crc_CalculateCRC32(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                  uint32_t Crc_StartValue32, uint8_t Crc_IsFirstCall) {
    uint32_t Crc = Crc_IsFirstCall ? 0xFFFFFFFFu : ~Crc_StartValue32;
    for(uint32_t byteindex = 0; byteindex < Crc_Length ; ++byteindex ) {
        Crc ^= *(Crc_DataPtr + byteindex);
        #ifdef CRC_TABLE_DISABLE
        for(uint8_t bit = 0; bit < 8u; ++bit){
            if((Crc & 1u) != 0)
                Crc = (Crc >> 1u) ^ 0xEDB88320u; //reflected 0x04C11DB7
            else
                Crc = (Crc >> 1u);
        }
        #else
            Crc = (Crc >> 8u) ^ CRC32_TABLE[(uint8_t)Crc]; 
        #endif
    }
    return ~Crc;
}


/*!  
 *  Specification        : [SWS_Crc_00056]
 *  CRC result width     : 32 bits
 *  Polynomial           : F4ACFB13h
 *  Initial value        : FFFFFFFFh
 *  Input data reflected : Yes
 *  Result data reflected: Yes
 *  XOR value            : FFFFFFFFh    
 */
uint32_t Crc_CalculateCRC32P4(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                    uint32_t Crc_StartValue32P4, uint8_t Crc_IsFirstCall) {
    uint32_t Crc = Crc_IsFirstCall ? 0xFFFFFFFFu : ~Crc_StartValue32P4;
    for(uint32_t byteindex = 0; byteindex < Crc_Length ; ++byteindex ) {
        Crc ^= *(Crc_DataPtr + byteindex);
        #ifdef CRC_TABLE_DISABLE
        for(uint8_t bit = 0; bit < 8u; ++bit){
            if((Crc & 1u) != 0)
                Crc = (Crc >> 1u) ^ 0xC8DF352Fu; //reflected 0xF4ACFB13
            else
                Crc = (Crc >> 1u);
        }
        #else
            Crc = (Crc >> 8u) ^ CRC32P4_TABLE[(uint8_t)Crc]; 
        #endif
    }
    return ~Crc;
}


/*!  
 *  Specification        : [SWS_Crc_00062]
 *  CRC result width     : 64 bits
 *  Polynomial           : 42F0E1EBA9EA3693h
 *  Initial value        : FFFFFFFFFFFFFFFFh
 *  Input data reflected : Yes
 *  Result data reflected: Yes
 *  XOR value            : FFFFFFFFFFFFFFFFh           
 */
uint64_t Crc_CalculateCRC64(const uint8_t *Crc_DataPtr, uint32_t Crc_Length,
                                  uint64_t Crc_StartValue64, uint8_t Crc_IsFirstCall) {
    uint64_t Crc = Crc_IsFirstCall ? 0xFFFFFFFFFFFFFFFFull: ~Crc_StartValue64;
    for(uint32_t byteindex = 0; byteindex < Crc_Length ; ++byteindex ) {
        Crc ^= *(Crc_DataPtr + byteindex);
        #ifdef CRC_TABLE_DISABLE
        for(uint8_t bit = 0; bit < 8u; ++bit){
            if((Crc & 1u) != 0)
                Crc = (Crc >> 1u) ^ 0xC96C5795D7870F42ull; //reflected 0x42F0E1EBA9EA3693
            else
                Crc = (Crc >> 1u);
        }
        #else
            Crc = (Crc >> 8ull) ^ CRC64_TABLE[(uint8_t)Crc]; 
        #endif
    }
    return ~Crc;                           
}