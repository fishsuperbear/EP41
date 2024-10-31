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
#include <utility>
#include "extend/e2e/protector_11.h"

namespace profile {
namespace profile11 {

Protector::Protector( Config cfg ) : config{std::move( cfg )}, counter{0U}, protectMutex{} {}

void Protector::Protect( crc::Buffer& buffer ) {
    if ( Profile11::IsBufferLengthValid( config, buffer ) ) {
        uint8_t currentCounter = [this]() -> uint8_t {
            std::lock_guard<std::mutex> lock{protectMutex};
            uint8_t                     tmpCounter{counter};
            IncrementCounter();
            return tmpCounter;
        }();
        WriteDataIdNibbleAndCounter( buffer, config.dataId, currentCounter );
        uint8_t computedCrc = Profile11::ComputeCrc( config, buffer );
        WriteCrc( buffer, computedCrc );
    } else {
        throw std::invalid_argument{"Length (" + std::to_string( buffer.size() ) +
                                    ") of the buffer is invalid."};
    }
}

void Protector::WriteDataIdNibbleAndCounter( crc::Buffer& buffer, uint16_t dataId,
                                             uint8_t currentCounter ) const noexcept {
    buffer[ config.crcOffset + 1U ] = 0U;
    if ( config.dataIdMode == DataIdMode::LOWER_12_BIT ) {
        buffer[ config.crcOffset + 1U ] += ( Profile11::CalculateDataIdNibble( dataId ) << 4U );
    }
    buffer[ config.crcOffset + 1U ] += ( currentCounter & 0x0FU );
}

void Protector::WriteCrc( crc::Buffer& buffer, uint8_t crc ) const noexcept {
    buffer[ config.crcOffset ] = crc;
}

void Protector::IncrementCounter() noexcept {
    if ( counter < Profile11::maxCounterValue ) {
        ++counter;
    } else {
        counter = 0x0U;
    }
}
}  // namespace profile11
}  // namespace profile
/* EOF */
