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
#include "extend/e2e/profile_07.h"

namespace profile {
namespace profile07 {

Config::Config( uint32_t curDataId, uint32_t curMaxDataLength, uint32_t curMinDataLength,
                uint32_t curMaxDeltaCounter, uint32_t curOffset )
    : dataId{curDataId}
    , maxDataLength{curMaxDataLength}
    , minDataLength{curMinDataLength}
    , maxDeltaCounter{curMaxDeltaCounter}
    , offset{curOffset} {
    static constexpr uint8_t minLength = 18U;
    if ( offset > ( maxDataLength - minLength ) ) {
        std::invalid_argument{"Wrong E2E configuration of profile 07: invalid offset value."};
    }
    if ( minDataLength < minLength || minDataLength > maxDataLength ) {
        std::invalid_argument{
            "Wrong E2E configuration of profile 07: min or max data length are invalid"};
    }
}

uint64_t Profile07::ComputeCrc( const Config&                config,
                                const crc::Buffer& buffer ) noexcept {
    using crc::CRC;
    uint64_t crc = 0xFFFFFFFFFFFFFFFFU;
    if ( config.offset > 0U ) {
        crc = CRC::CalculateCRC64( crc::BufferView{buffer, 0U, config.offset} );
        crc = CRC::CalculateCRC64(
            crc::BufferView{buffer, config.offset + 8U, buffer.size()}, false, crc );
    } else {
        crc = CRC::CalculateCRC64(
            crc::BufferView{buffer, config.offset + 8U, buffer.size()}, true, crc );
    }
    return crc;
}

bool Profile07::IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept {
    return ( ( config.minDataLength <= buffer.size() ) &&
             ( buffer.size() <= config.maxDataLength ) );
}
}  // namespace profile07
}  // namespace profile
/* EOF */
