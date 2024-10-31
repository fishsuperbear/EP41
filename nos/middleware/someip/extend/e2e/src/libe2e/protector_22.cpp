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
#include "extend/e2e/protector_22.h"

namespace profile {
namespace profile22 {

Protector::Protector( Config cfg ) : config{std::move( cfg )}, counter{0U}, protectMutex{} {}

void Protector::Protect( crc::Buffer& buffer ) {
    if ( Profile22::IsBufferLengthValid( config, buffer ) ) {
        uint8_t currentCounter = [this]() -> uint8_t {
            std::lock_guard<std::mutex> lock{protectMutex};
            uint8_t                     tmpCounter{counter};
            IncrementCounter();
            return tmpCounter;
        }();
        WriteCounter( buffer, currentCounter );
        uint8_t computedCrc = Profile22::ComputeCrc( config, buffer, currentCounter );
        WriteCrc( buffer, computedCrc );
    } else {
        throw std::invalid_argument{"Length of the buffer is invalid."};
    }
}

void Protector::WriteCrc( crc::Buffer& buffer, uint8_t crc ) const noexcept {
    buffer[ config.offset ] = crc;
}

void Protector::WriteCounter( crc::Buffer& buffer, uint8_t currentCounter ) const
    noexcept {
    buffer[ config.offset + 1U ] =
        ( buffer[ config.offset + 1U ] & 0xF0U ) | static_cast<uint8_t>( currentCounter & 0x0FU );
}

void Protector::IncrementCounter() noexcept {
    if ( counter < Profile22::maxCounterValue ) {
        ++counter;
    } else {
        counter = 0x0U;
    }
}
}  // namespace profile22
}  // namespace profile
/* EOF */
