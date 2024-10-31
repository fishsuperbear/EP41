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
#include "extend/e2e/protector_07.h"

namespace profile {
namespace profile07 {

Protector::Protector( Config cfg ) : config{std::move( cfg )}, counter{0U}, protectMutex{} {}

void Protector::Protect( crc::Buffer& buffer ) {
    if ( Profile07::IsBufferLengthValid( config, buffer ) ) {
        WriteLength( buffer, static_cast<uint32_t>( buffer.size() ) );
        uint32_t currentCounter = [this]() -> uint32_t {
            std::lock_guard<std::mutex> lock( protectMutex );
            uint32_t                    tmpCounter{counter};
            IncrementCounter();
            return tmpCounter;
        }();
        WriteCounter( buffer, currentCounter );
        WriteDataId( buffer, config.dataId );
        uint64_t computedCrc = Profile07::ComputeCrc( config, buffer );
        WriteCrc( buffer, computedCrc );
    } else {
        throw std::out_of_range{"Length of the buffer is invalid."};
    }
}

void Protector::WriteLength( crc::Buffer& buffer, uint32_t length ) const noexcept {
    buffer[ config.offset + 8U ]  = static_cast<uint8_t>( ( length & 0xFF000000U ) >> 24U );
    buffer[ config.offset + 9U ]  = static_cast<uint8_t>( ( length & 0xFF0000U ) >> 16U );
    buffer[ config.offset + 10U ] = static_cast<uint8_t>( ( length & 0xFF00U ) >> 8U );
    buffer[ config.offset + 11U ] = static_cast<uint8_t>( length & 0xFFU );
}

void Protector::WriteCounter( crc::Buffer& buffer, uint32_t currentCounter ) const
    noexcept {
    buffer[ config.offset + 12U ] =
        static_cast<uint8_t>( ( currentCounter & 0xFF000000U ) >> 24U );
    buffer[ config.offset + 13U ] = static_cast<uint8_t>( ( currentCounter & 0xFF0000U ) >> 16U );
    buffer[ config.offset + 14U ] = static_cast<uint8_t>( ( currentCounter & 0xFF00U ) >> 8U );
    buffer[ config.offset + 15U ] = static_cast<uint8_t>( currentCounter & 0xFFU );
}

void Protector::WriteDataId( crc::Buffer& buffer, uint32_t dataId ) const noexcept {
    buffer[ config.offset + 16U ] = static_cast<uint8_t>( ( dataId & 0xFF000000U ) >> 24U );
    buffer[ config.offset + 17U ] = static_cast<uint8_t>( ( dataId & 0xFF0000U ) >> 16U );
    buffer[ config.offset + 18U ] = static_cast<uint8_t>( ( dataId & 0xFF00U ) >> 8U );
    buffer[ config.offset + 19U ] = static_cast<uint8_t>( dataId & 0xFFU );
}

void Protector::WriteCrc( crc::Buffer& buffer, uint64_t crc ) const noexcept {
    buffer[ config.offset ] = static_cast<uint8_t>( ( crc & 0xFF00000000000000U ) >> 56U );
    buffer[ config.offset + 1U ] = static_cast<uint8_t>( ( crc & 0xFF000000000000U ) >> 48U );
    buffer[ config.offset + 2U ] = static_cast<uint8_t>( ( crc & 0xFF0000000000U ) >> 40U );
    buffer[ config.offset + 3U ] = static_cast<uint8_t>( ( crc & 0xFF00000000U ) >> 32U );
    buffer[ config.offset + 4U ] = static_cast<uint8_t>( ( crc & 0xFF000000U ) >> 24U );
    buffer[ config.offset + 5U ] = static_cast<uint8_t>( ( crc & 0xFF0000U ) >> 16U );
    buffer[ config.offset + 6U ] = static_cast<uint8_t>( ( crc & 0xFF00U ) >> 8U );
    buffer[ config.offset + 7U ] = static_cast<uint8_t>( crc & 0xFFU );
}

void Protector::IncrementCounter() noexcept { ++counter; }
}  // namespace profile07
}  // namespace profile
/* EOF */
