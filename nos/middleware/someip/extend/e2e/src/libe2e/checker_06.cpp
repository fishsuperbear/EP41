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

#include "extend/e2e/checker_06.h"
#include <utility>

namespace profile {
namespace profile06 {

Checker::Checker( Config cfg ) : config{std::move( cfg )}, counter{0xFFU}, checkMutex{} {}

void Checker::Check( const crc::Buffer&            buffer,
                     E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) noexcept {
    if ( buffer.empty() ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kNoNewData;
    } else if ( !Profile06::IsBufferLengthValid( config, buffer ) ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kError;
    } else {
        uint8_t                     receivedCounter = ReadCounter( buffer );
        std::lock_guard<std::mutex> lock{checkMutex};
        ProfileCheckStatus          checkStatus =
            DoChecks( static_cast<uint16_t>( buffer.size() ), ReadLength( buffer ), counter,
                      receivedCounter, Profile06::ComputeCrc( config, buffer ), ReadCrc( buffer ) );
        if ( checkStatus != ProfileCheckStatus::kError ) {
            counter = receivedCounter;
        }
        genericProfileCheckStatus = MapProfileCheckStatusToCheckStatus( checkStatus );
    }
}

uint16_t Checker::ReadLength( const crc::Buffer& buffer ) const noexcept {
    return ( ( static_cast<uint16_t>( buffer[ config.offset + 2U ] ) << 8U ) |
             ( static_cast<uint16_t>( buffer[ config.offset + 3U ] ) ) );
}

uint8_t Checker::ReadCounter( const crc::Buffer& buffer ) const noexcept {
    return buffer[ config.offset + 4U ];
}

uint16_t Checker::ReadCrc( const crc::Buffer& buffer ) const noexcept {
    return ( ( static_cast<uint16_t>( buffer[ config.offset ] ) << 8U ) |
             ( static_cast<uint16_t>( buffer[ config.offset + 1U ] ) ) );
}

ProfileCheckStatus Checker::DoChecks( uint16_t length, uint16_t receivedLength,
                                      uint8_t currentCounter, uint8_t receivedCounter,
                                      uint16_t computedCRC, uint16_t receivedCRC ) noexcept {
    if ( ( computedCRC == receivedCRC ) && ( length == receivedLength ) ) {
        return CheckCounter( receivedCounter, currentCounter );
    }
    return ProfileCheckStatus::kError;
}

ProfileCheckStatus Checker::CheckCounter( uint8_t receivedCounter,
                                          uint8_t currentCounter ) noexcept {
    const uint8_t delta{static_cast<uint8_t>( receivedCounter - currentCounter )};
    if ( delta <= config.maxDeltaCounter ) {
        if ( delta == 0U ) {
            return ProfileCheckStatus::kRepeated;
        } else if ( delta == 1U ) {
            return ProfileCheckStatus::kOk;
        } else {
            return ProfileCheckStatus::kOkSomeLost;
        }
    } else {
        return ProfileCheckStatus::kWrongSequence;
    }
}
}  // namespace profile06
}  // namespace profile
/* EOF */
