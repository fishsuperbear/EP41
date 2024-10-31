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
#include "extend/e2e/profile_22.h"

namespace profile {
namespace profile22 {

Config::Config( uint16_t curDataId, const std::array<std::uint8_t, 16>& curDataIdList,
                uint16_t curDataLength, uint8_t curMaxDeltaCounter, uint16_t curOffset )
    : dataId{curDataId}
    , dataIdList{curDataIdList}
    , dataLength{curDataLength}
    , maxDeltaCounter{curMaxDeltaCounter}
    , offset{curOffset} {
    static constexpr uint8_t maxLength{255U};
    if ( dataLength + 1U > maxLength ) {
        throw std::invalid_argument{
            "Wrong E2E configuration of profile 22: data length is greater than 256."};
    }
}

uint8_t Profile22::ComputeCrc( const Config& config, const crc::Buffer& buffer,
                               uint8_t counter ) noexcept {
    using crc::CRC;
    uint8_t computedCrc;
    if ( config.offset > 0U ) {
        computedCrc = CRC::CalculateCRC8H2F( crc::BufferView{buffer, config.offset} );
        if ( config.dataLength > config.offset ) {
            computedCrc = CRC::CalculateCRC8H2F(
                crc::BufferView{buffer, static_cast<size_t>( config.offset + 1U ),
                                          config.dataLength},
                false, computedCrc );
        }
    } else {
        computedCrc =
            CRC::CalculateCRC8H2F( crc::BufferView{buffer, 1U, config.dataLength} );
    }

    crc::Buffer tmpBuf{static_cast<uint8_t>( config.dataIdList[ counter ] )};
    computedCrc = CRC::CalculateCRC8H2F( crc::BufferView{tmpBuf}, false, computedCrc );
    return computedCrc;
}

bool Profile22::IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept {
    return ( buffer.size() == config.dataLength );
}
}  // namespace profile22
}  // namespace profile
   /* EOF */