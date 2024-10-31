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
#ifndef E2E_INCLUDE_E2E_PROFILE07_H_
#define E2E_INCLUDE_E2E_PROFILE07_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "extend/crc/buffer.h"

namespace profile {
namespace profile07 {

/// @brief Configuration of transmitted Data for E2E Profile 07
struct Config {
    /// @brief A system-unique identifier of the Data
    uint32_t dataId;

    /// @brief Maximal length of Data. E2E checks that Length is =< MinDataLength.
    uint32_t maxDataLength;

    /// @brief Minimal length of Data. E2E checks that Length is >= MinDataLength.
    uint32_t minDataLength;

    /// @brief Maximum allowed gap between two counter values of two consecutively received valid
    /// Data
    uint32_t maxDeltaCounter;

    /// @brief Bit offset of the first bit of the E2E header from the beginning of the Data (bit
    /// numbering: bit 0 is the least important).
    uint32_t offset;

    Config() = delete;

    /// @brief Constructs configuration object for E2E profile from supplied parameters
    Config( uint32_t dataId, uint32_t maxDataLength, uint32_t minDataLength,
            uint32_t maxDeltaCounter, uint32_t offset );

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

/// @brief Specific values and routines for profile 07
class Profile07 {
   public:
    /// @brief length of E2E header
    static constexpr uint16_t headerLength{20U};

    /// @brief Compute CRC over buffer using configuration values provided in config
    ///
    /// @param config profile configuration
    /// @param buffer CRC is calculated done over this data
    ///
    /// @return computed CRC as uint64_t value
    ///
    /// @uptrace{SWS_CM_90402}
    static uint64_t ComputeCrc( const Config&                config,
                                const crc::Buffer& buffer ) noexcept;

    /// @brief Checks if buffer length is in range (minDataLength...maxDataLength)
    ///
    /// @return true if data is in range, false - otherwise
    static bool IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept;
};

}  // namespace profile07
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_PROFILE07_H_
/* EOF */
