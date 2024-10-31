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
#ifndef E2E_INCLUDE_E2E_PROFILE11_H_
#define E2E_INCLUDE_E2E_PROFILE11_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include "extend/crc/buffer.h"

namespace profile {
namespace profile11 {

/// @brief Supported inclusion modes to include the implicit two-byte Data ID in the one-byte CRC
///
/// @uptrace{SWS_CM_90403}
enum class DataIdMode : uint8_t {
    ALL_16_BIT,   ///< Two bytes are included in the CRC (double ID configuration).
    LOWER_12_BIT  ///< The low byte is included in the implicit CRC calculation, the low nibble of
                  ///< the high byte is
    /// transmitted along with the data (i.e. it is explicitly included), the high nibble of the
    /// high byte is not used. This is applicable for the IDs up to 12 bits.
};

/// @brief Configuration of transmitted Data for E2E Profile 11
struct Config {
    /// @brief Bit offset of Counter in MSB first order
    uint16_t counterOffset;

    /// @brief Bit offset of CRC (i.e. since *Data) in MSB first order
    uint16_t crcOffset;

    /// @brief A unique numerical identifier for the referenced event or field notifier that is
    /// included in the CRC calculation
    uint16_t dataId;

    /// @brief the inclusion mode that is used to include the implicit two-byte Data ID in the
    /// one-byte CRC
    DataIdMode dataIdMode;

    /// @brief Bit offset of the low nibble of the high byte of Data ID. This parameter is used only
    /// if dataIdMode = LOWER_12_BIT (otherwise it is ignored)
    uint16_t dataIdNibbleOffset;

    /// @brief Length of data, in bits
    uint16_t dataLength;

    /// @brief Maximum allowed difference between two counter values of two consecutively received
    /// valid messages.
    uint8_t maxDeltaCounter;

    Config() = delete;

    /// @brief Constructs configuration object for E2E profile from supplied parameters
    Config( uint16_t counterOffset, uint16_t crcOffset, uint16_t dataId, DataIdMode dataIdMode,
            uint16_t dataIdNibbleOffset, uint16_t dataLength, uint8_t maxDeltaCounter );

    /// @brief Default copy constructor
    Config( const Config& config ) = default;

    /// @brief Default move constructor
    Config( Config&& config ) noexcept = default;

    /// @brief Default assignment operator
    Config& operator=( const Config& config ) = default;

    /// @brief Default move-assignment operator
    Config& operator=( Config&& config ) noexcept = default;

    /// @brief Default destructor
    ~Config() noexcept = default;
};

/// @brief Specific values and routines for profile 11
class Profile11 {
   public:
    /// @brief length of E2E header
    static constexpr uint16_t headerLength{2U};

    /// @brief maximum value of counter
    static constexpr uint8_t maxCounterValue{0x0EU};

    /// @brief default data length supported by profile
    static constexpr uint16_t dataLength{8U};

    // Constants below defined according PRS_E2EProtocol_00540 and PRS_E2EProtocol_00541

    /// @brief Default bit offset of CRC
    static constexpr uint16_t crcOffset{0U};

    /// @brief Default offset of Counter
    static constexpr uint16_t counterOffset{8U};

    /// @brief Default offset of the low nibble of the high byte of Data ID
    static constexpr uint16_t dataIdNibbleOffset{12U};

    /// @brief Compute CRC over buffer using configuration values provided in config
    ///
    /// @param config profile configuration
    /// @param buffer CRC is calculated done over this data
    ///
    /// @return computed CRC as uint8_t value
    ///
    /// @uptrace{SWS_CM_90402}
    static uint8_t ComputeCrc( const Config& config, const crc::Buffer& buffer ) noexcept;

    /// @brief Extracts nibble from Data ID
    ///
    /// @param dataId A unique numerical identifier for the referenced event or field notifier
    ///
    /// @return nibble value
    static uint8_t CalculateDataIdNibble( uint16_t dataId ) noexcept;

    /// @brief Checks if buffer length is in range (minDataLength...maxDataLength)
    ///
    /// @return true if data is in range, false - otherwise
    static bool IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept;
};

}  // namespace profile11
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_PROFILE11_H_
/* EOF */
