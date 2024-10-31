/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#ifndef E2E_INCLUDE_CRC_CRC_H_
#define E2E_INCLUDE_CRC_CRC_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include "buffer.h"

/// The CRC calculation library
///
/// Library contains the following routines for CRC calculation:
/// - CRC8: SAEJ1850
/// - CRC8H2F: CRC8 0x2F polynomial
/// - CRC16
/// - CRC32
/// - CRC32P4: CRC32 0x1F4ACFB13 polynomial
/// - CRC64: CRC-64-ECMA
///
/// For all routines (CRC8, CRC8H2F, CRC16, CRC32, CRC32P4 and CRC64) table based calculation method
/// is implemented
///
/// All routines are re-entrant and can be used by multiple applications at the same time.
///
/// Hardware CRC calculation may be supported by some devices in the future.

namespace crc {

/// @brief CRC calculation routines
class CRC {
   public:
    /// @brief initial value for CRC32P4-calculation
    static constexpr uint32_t startValueCrc32P4{0xFFFFFFFFU};

    /// @brief initial value for CRC32-calculation
    static constexpr uint32_t startValueCrc32{0xFFFFFFFFU};

    /// @brief initial value for CRC16-calculation
    static constexpr uint16_t startValueCrc16{0xFFFFU};

    /// @brief initial value for CRC64-calculation
    static constexpr uint64_t startValueCrc64{0xFFFFFFFFFFFFFFFFU};

    /// @brief initial value for CRC8-calculation
    static constexpr uint8_t startValueCrc8{0xFFU};

    /// @brief initial value for CRC8H2F-calculation
    static constexpr uint8_t startValueCrc8H2F{0xFFU};

    /// @brief This method does a CRC32 calculation on the data
    ///        defined in the input buffer using the polynomial 0xF4ACFB13.
    ///        - Width        = 32
    ///        - Poly         = 0xF4ACFB13
    ///        - XorIn        = 0xFFFFFFFF
    ///        - ReflectIn    = true
    ///        - XorOut       = 0xFFFFFFFF
    ///        - ReflectOut   = true
    ///        - Algorithm    = table-driven
    ///
    /// @param bufferView   Buffer containing the data bytes and length of data
    /// @param isFirstCall true:  first call in a sequence or individual CRC calculation
    ///                     false: subsequent call in a call sequence
    /// @param startValue   Start value when the algorihtms starts
    /// @return             32 bit result of CRC calculation in range
    /// @retval             0,...,2^32-1
    ///
    /// @uptrace{SWS_Crc_00058}
    static uint32_t CalculateCRC32P4( BufferView bufferView, bool isFirstCall = true,
                                      uint32_t startValue = startValueCrc32P4 ) noexcept;

    /// @brief This method does a CRC32 (IEEE-802.3 CRC32 Ethernet Standard) calculation on the data
    ///        defined in the input buffer using the polynomial 0x04C11DB7.
    ///         - Width        = 32
    ///         - Poly         = 0x04C11DB7
    ///         - XorIn        = 0xFFFFFFFF
    ///         - ReflectIn    = true
    ///         - XorOut       = 0xFFFFFFFF
    ///         - ReflectOut   = true
    ///         - Algorithm    = table-driven
    ///
    /// @param bufferView   Buffer containing the data bytes and length of data
    /// @param isFirstCall true:  first call in a sequence or individual CRC calculation
    ///                     false: subsequent call in a call sequence
    /// @param startValue   Start value when the algorihtms starts
    /// @return             32 bit result of CRC calculation
    /// @retval             0,...,2^32-1
    ///
    /// @uptrace{SWS_Crc_00058}
    static uint32_t CalculateCRC32( BufferView bufferView, bool isFirstCall = true,
                                    uint32_t startValue = startValueCrc32 ) noexcept;

    /// @brief This method does a CRC16 calculation on the data
    ///        defined in the input buffer using the polynomial 0x1021.
    ///         - Width        = 16
    ///         - Poly         = 0x1021
    ///         - XorIn        = 0xffff
    ///         - ReflectIn    = False
    ///         - XorOut       = 0x0000
    ///         - ReflectOut   = False
    ///         - Algorithm    = table-driven
    ///
    /// @param bufferView   Buffer containing the data bytes and length of data
    /// @param isFirstCall true:  first call in a sequence or individual CRC calculation
    ///                     false: subsequent call in a call sequence
    /// @param startValue   Start value when the algorihtms starts
    /// @return             16 bit result of CRC calculation
    /// @retval             0,...,2^16-1
    ///
    /// @uptrace{SWS_Crc_00019}
    static uint16_t CalculateCRC16( BufferView bufferView, bool isFirstCall = true,
                                    uint16_t startValue = startValueCrc16 ) noexcept;

    /// @brief This method does a CRC64 calculation on the data
    ///        defined in the input buffer using the polynomial 0x42F0E1EBA9EA3693.
    ///         - Width        = 64
    ///         - Poly         = 0x42F0E1EBA9EA3693
    ///         - XorIn        = 0xffffffff
    ///         - ReflectIn    = True
    ///         - XorOut       = 0xffffffff
    ///         - ReflectOut   = True
    ///         - Algorithm    = bit-by-bit-fast
    ///
    /// @param bufferView   Buffer containing the data bytes and length of data
    /// @param isFirstCall true:  first call in a sequence or individual CRC calculation
    ///                     false: subsequent call in a call sequence
    /// @param startValue   Start value when the algorihtms starts
    /// @return             64 bit result of CRC calculation
    /// @retval             0,...,2^64-1
    ///
    /// @uptrace{SWS_Crc_00061}
    static uint64_t CalculateCRC64( BufferView bufferView, bool isFirstCall = true,
                                    uint64_t startValue = startValueCrc64 ) noexcept;

    /// @brief This method does a CRC8 (SAE J1850) calculation on the data
    ///        defined in the input buffer using the polynomial 0x1D.
    ///         - Width        = 8
    ///         - Poly         = 0x2f
    ///         - XorIn        = 0xff
    ///         - ReflectIn    = False
    ///         - XorOut       = 0xff
    ///         - ReflectOut   = False
    ///         - Algorithm    = table-driven
    ///
    /// @param bufferView   Buffer containing the data bytes and length of data
    /// @param isFirstCall true:  first call in a sequence or individual CRC calculation
    ///                     false: subsequent call in a call sequence
    /// @param startValue   Start value when the algorihtms starts
    /// @return             8 bit result of CRC calculation
    /// @retval             0,...,255
    ///
    /// @uptrace{SWS_Crc_00031}
    static uint8_t CalculateCRC8( BufferView bufferView, bool isFirstCall = true,
                                  uint8_t startValue = startValueCrc8 ) noexcept;

    /// @brief This method does a CRC8H2F calculation on the data
    ///        defined in the input buffer using the polynomial 0x2F.
    ///         - Width        = 8
    ///         - Poly         = 0x2f
    ///         - XorIn        = 0xff
    ///         - ReflectIn    = False
    ///         - XorOut       = 0xff
    ///         - ReflectOut   = False
    ///         - Algorithm    = table-driven
    ///
    /// @param bufferView   Buffer containing the data bytes and length of data
    /// @param isFirstCall true:  first call in a sequence or individual CRC calculation
    ///                     false: subsequent call in a call sequence
    /// @param startValue   Start value when the algorihtms starts
    /// @return             8 bit result of CRC calculation
    /// @retval             0,...,255
    ///
    /// @uptrace{SWS_Crc_00043}
    static uint8_t CalculateCRC8H2F( BufferView bufferView, bool isFirstCall = true,
                                     uint8_t startValue = startValueCrc8H2F ) noexcept;

   private:
    /// @brief XOR values for the different CRC-calculations
    static constexpr uint32_t xorValueCrc32P4{0xFFFFFFFFU};
    static constexpr uint32_t xorValueCrc32{0xFFFFFFFFU};
    static constexpr uint16_t xorValueCrc16{0x0000U};
    static constexpr uint64_t xorValueCrc64{0xFFFFFFFFFFFFFFFFU};
    static constexpr uint8_t  xorValueCrc8{0xFFU};
    static constexpr uint8_t  xorValueCrc8H2F{0xFFU};

    static uint32_t Reflect( uint32_t crc ) noexcept;
    static uint64_t Reflect( uint64_t crc ) noexcept;
};

}  // namespace crc

#endif  // E2E_INCLUDE_CRC_CRC_H_
        /* EOF */