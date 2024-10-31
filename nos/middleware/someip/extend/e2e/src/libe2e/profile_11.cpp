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
#include <stdexcept>
#include "extend/crc/crc.h"
#include "extend/e2e/profile_11.h"

namespace profile {
namespace profile11 {

Config::Config( uint16_t curCounterOffset, uint16_t curCrcOffset, uint16_t curDataId,
                DataIdMode curDataIdMode, uint16_t curDataIdNibbleOffset, uint16_t curDataLength,
                uint8_t curMaxDeltaCounter )
    : counterOffset{curCounterOffset}
    , crcOffset{curCrcOffset}
    , dataId{curDataId}
    , dataIdMode{curDataIdMode}
    , dataIdNibbleOffset{curDataIdNibbleOffset}
    , dataLength{curDataLength}
    , maxDeltaCounter{curMaxDeltaCounter} {
    /// \uptrace{SWS_CM_90402}
    static constexpr uint8_t maxLength{240U};
    if ( dataLength > maxLength ) {
        throw std::invalid_argument{
            "Wrong E2E configuration of profile 11: data length is greater than 240."};
    }
}

uint8_t Profile11::ComputeCrc( const Config&                config,
                               const crc::Buffer& buffer ) noexcept {
    using crc::CRC;

    uint8_t computedCrc;
    if ( config.dataIdMode == DataIdMode::LOWER_12_BIT ) {
        crc::Buffer tmpBuf{static_cast<uint8_t>( config.dataId & 0x00FFU )};
        computedCrc = CRC::CalculateCRC8( crc::BufferView{tmpBuf} );
        tmpBuf      = crc::Buffer{0U};
        computedCrc = CRC::CalculateCRC8( crc::BufferView{tmpBuf}, false, computedCrc );
    } else {
        crc::Buffer tmpBuf{static_cast<uint8_t>( config.dataId & 0x00FFU )};
        computedCrc = CRC::CalculateCRC8( crc::BufferView{tmpBuf} );
        tmpBuf = crc::Buffer{static_cast<uint8_t>( ( config.dataId >> 8 ) & 0xFFU )};
        computedCrc = CRC::CalculateCRC8( crc::BufferView{tmpBuf}, false, computedCrc );
    }

    // FIXME: is real data length is config.dataLength or config.dataLength + config.crcOffset?
    if ( config.crcOffset > 0U ) {
        computedCrc = CRC::CalculateCRC8( crc::BufferView{buffer, config.crcOffset},
                                          false, computedCrc );
        if ( config.dataLength > config.crcOffset ) {
            computedCrc = CRC::CalculateCRC8(
                crc::BufferView{buffer, static_cast<uint32_t>( config.crcOffset ) + 1U,
                                          config.dataLength},
                false, computedCrc );
        }
    } else {
        computedCrc = CRC::CalculateCRC8( crc::BufferView{buffer, 1U, config.dataLength},
                                          false, computedCrc );
    }
    return ( computedCrc ^ 0xFF );
}

uint8_t Profile11::CalculateDataIdNibble( uint16_t dataId ) noexcept {
    return static_cast<uint8_t>( ( dataId & 0x0F00U ) >> 8U );
}

bool Profile11::IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept {
    return ( buffer.size() == config.dataLength );
}
}  // namespace profile11
}  // namespace profile
/* EOF */
