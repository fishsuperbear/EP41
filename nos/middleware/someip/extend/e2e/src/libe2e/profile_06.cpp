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
#include "extend/e2e/profile_06.h"
#include "extend/crc/crc.h"

namespace profile {
namespace profile06 {

Config::Config( uint16_t curDataId, uint16_t curMinDataLength, uint16_t curMaxDataLength,
                uint16_t curMaxDeltaCounter, uint16_t curOffset )
    : dataId{curDataId}
    , minDataLength{curMinDataLength}
    , maxDataLength{curMaxDataLength}
    , maxDeltaCounter{curMaxDeltaCounter}
    , offset{curOffset} {
    if ( offset > ( maxDataLength - Profile06::headerLength ) ) {
        throw std::invalid_argument{"Wrong configuration of E2E profile 6: Invalid offset"};
    }
    if ( minDataLength < Profile06::headerLength ||
         minDataLength > Profile06::maxCrcProtectedDataLength ) {
        throw std::invalid_argument{"Wrong configuration of E2E profile 6: Invalid minDataLength"};
    }
    if ( maxDataLength > Profile06::maxCrcProtectedDataLength ) {
        throw std::invalid_argument{"Wrong configuration of E2E profile 6: Invalid maxDataLength"};
    }
    if ( maxDataLength < minDataLength ) {
        throw std::invalid_argument{
            "Wrong configuration of E2E profile 6: maxDataLength less than minDataLength"};
    }
}

uint16_t Profile06::ComputeCrc( const Config&                config,
                                const crc::Buffer& buffer ) noexcept {
    using crc::CRC;
    uint16_t crc;
    if ( config.offset > 0U ) {
        crc = CRC::CalculateCRC16( crc::BufferView{buffer, config.offset} );
        crc = CRC::CalculateCRC16(
            crc::BufferView{buffer, static_cast<uint32_t>( config.offset ) + 2U,
                                      buffer.size()},
            false, crc );
    } else {
        crc = CRC::CalculateCRC16( crc::BufferView{
            buffer, static_cast<uint32_t>( config.offset ) + 2U, buffer.size()} );
    }

    crc::Buffer tmpBuffer;
    tmpBuffer.push_back( static_cast<uint8_t>( ( config.dataId & 0xFF00U ) >> 8U ) );
    crc = CRC::CalculateCRC16( crc::BufferView{tmpBuffer}, false, crc );

    tmpBuffer.clear();
    tmpBuffer.push_back( static_cast<uint8_t>( config.dataId & 0xFFU ) );
    crc = CRC::CalculateCRC16( crc::BufferView{tmpBuffer}, false, crc );
    return crc;
}

/** \uptrace{SWS_E2E_00356} */
bool Profile06::IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept {
    return ( config.minDataLength <= buffer.size() && buffer.size() <= config.maxDataLength );
}
}  // namespace profile06
}  // namespace profile
   /* EOF */