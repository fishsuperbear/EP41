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
#ifndef E2E_INCLUDE_E2E_PROFILE05_H_
#define E2E_INCLUDE_E2E_PROFILE05_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include "extend/crc/buffer.h"

namespace profile {
namespace profile05 {
/// @brief Configuration of transmitted Data for E2E Profile 05
struct Config {
    /// @brief A system-unique identifier of the Data
    uint16_t dataId;

    /// @brief Length of Data
    uint16_t dataLength;

    /// @brief Maximum allowed gap between two counter values of two consecutively received valid
    /// Data.
    uint8_t maxDeltaCounter;

    /// @brief Bit offset of the first bit of the E2E header from the beginning of the Data (bit
    /// numbering: bit 0 is the least important).
    uint16_t offset;

#ifndef E2E_DEVELOPMENT
    Config() = delete;
#else
    Config() = default;
#endif

    /// @brief Constructs configuration object for E2E profile from supplied parameters
    Config( uint16_t dataId, uint16_t dataLength, uint8_t maxDeltaCounter, uint16_t offset );

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

/// @brief Specific values and routines for profile 05
class Profile05 {
   public:
    /// @brief length of E2E header
    static constexpr uint16_t headerLength{3U};
    /// @brief maximum length of the data that can be protected by CRC type supported by profile
    static constexpr uint16_t maxCrcProtectedDataLength{4096U};
    /// @brief fixed data length value for profile
    static constexpr uint16_t dataLength = 8U;

    /// @brief Compute CRC over buffer using configuration values provided in config
    ///
    /// @param config profile configuration
    /// @param buffer CRC is calculated done over this data
    ///
    /// @return computed CRC as uint16_t value
    ///
    /// @uptrace{SWS_CM_90402}
    static uint16_t ComputeCrc( const Config&                config,
                                const crc::Buffer& buffer ) noexcept;

    /// @brief Checks if buffer length is in range (minDataLength...maxDataLength)
    ///
    /// @return true if data is in range, false - otherwise
    static bool IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept;
};

}  // namespace profile05
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_PROFILE05_H_
/* EOF */
