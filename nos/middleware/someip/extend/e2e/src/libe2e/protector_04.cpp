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
#include "extend/e2e/protector_04.h"

namespace profile {
namespace profile04 {

Protector::Protector( Config cfg ) : config{std::move( cfg )}, counter{0U}, protectMutex{} {}

/** \uptrace{SWS_CM_90433} */
void Protector::Protect( crc::Buffer& buffer ) {
    if ( Profile04::IsBufferLengthValid( config, buffer ) ) {
        WriteLength( buffer, static_cast<uint16_t>( buffer.size() ) );
        uint16_t currentCounter = [this]() -> uint16_t {
            std::lock_guard<std::mutex> lock( protectMutex );
            uint16_t                    tmpValue{counter};
            IncrementCounter();
            return tmpValue;
        }();
        WriteCounter( buffer, currentCounter );
        WritedataId( buffer, config.dataId );
        uint32_t computedCRC = Profile04::ComputeCrc( config, buffer );
        WriteCrc( buffer, computedCRC );
    } else {
        throw std::out_of_range{"Length of the buffer is invalid."};
    }
}

/** \uptrace{SWS_CM_90433} */
void Protector::WriteLength( crc::Buffer& buffer, uint16_t length ) noexcept {
    buffer[ config.offset ]      = static_cast<uint8_t>( length >> 8U );
    buffer[ config.offset + 1U ] = static_cast<uint8_t>( length );
}

/** \uptrace{SWS_CM_90433} */
void Protector::WriteCounter( crc::Buffer& buffer, uint16_t currentCounter ) noexcept {
    buffer[ config.offset + 2U ] = static_cast<uint8_t>( currentCounter >> 8U );
    buffer[ config.offset + 3U ] = static_cast<uint8_t>( currentCounter );
}

/** \uptrace{SWS_CM_90433} */
void Protector::WritedataId( crc::Buffer& buffer, uint32_t dataId ) noexcept {
    buffer[ config.offset + 4U ] = static_cast<uint8_t>( dataId >> 24U );
    buffer[ config.offset + 5U ] = static_cast<uint8_t>( dataId >> 16U );
    buffer[ config.offset + 6U ] = static_cast<uint8_t>( dataId >> 8U );
    buffer[ config.offset + 7U ] = static_cast<uint8_t>( dataId );
}

/** \uptrace{SWS_CM_90433} */
void Protector::WriteCrc( crc::Buffer& buffer, uint32_t computedCRC ) noexcept {
    buffer[ config.offset + 8U ]  = static_cast<uint8_t>( computedCRC >> 24U );
    buffer[ config.offset + 9U ]  = static_cast<uint8_t>( computedCRC >> 16U );
    buffer[ config.offset + 10U ] = static_cast<uint8_t>( computedCRC >> 8U );
    buffer[ config.offset + 11U ] = static_cast<uint8_t>( computedCRC );
}

/** \uptrace{SWS_CM_90433} */
void Protector::IncrementCounter() noexcept { ++counter; }
}  // namespace profile04
}  // namespace profile
/* EOF */
