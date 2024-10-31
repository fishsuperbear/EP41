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
#include "extend/e2e/protector_06.h"

namespace profile {
namespace profile06 {

Protector::Protector( Config cfg ) : config{std::move( cfg )}, counter{0U}, protectMutex{} {}

void Protector::Protect( crc::Buffer& buffer ) {
    if ( Profile06::IsBufferLengthValid( config, buffer ) ) {
        WriteLength( buffer, static_cast<uint16_t>( buffer.size() ) );
        uint8_t currentCounter = [this]() -> uint8_t {
            std::lock_guard<std::mutex> lock( protectMutex );
            uint8_t                     tmpValue{counter};
            IncrementCounter();
            return tmpValue;
        }();
        WriteCounter( buffer, currentCounter );
        uint16_t computedCRC{Profile06::ComputeCrc( config, buffer )};
        WriteCrc( buffer, computedCRC );
    } else {
        throw std::out_of_range{"Length of the buffer is invalid."};
    }
}

void Protector::WriteLength( crc::Buffer& buffer, uint16_t length ) noexcept {
    buffer[ config.offset + 2U ] = static_cast<uint8_t>( ( length & 0xFF00U ) >> 8U );
    buffer[ config.offset + 3U ] = static_cast<uint8_t>( length & 0xFFU );
}

void Protector::WriteCounter( crc::Buffer& buffer, uint8_t currentCounter ) noexcept {
    buffer[ config.offset + 4U ] = currentCounter;
}

void Protector::WriteCrc( crc::Buffer& buffer, uint16_t computedCRC ) noexcept {
    buffer[ config.offset ]      = static_cast<uint8_t>( ( computedCRC & 0xFF00U ) >> 8U );
    buffer[ config.offset + 1U ] = static_cast<uint8_t>( computedCRC & 0xFFU );
}

void Protector::IncrementCounter() noexcept { ++counter; }
}  // namespace profile06
}  // namespace profile
/* EOF */
