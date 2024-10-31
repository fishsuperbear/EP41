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
#include "extend/e2e/checker_04.h"
#include <utility>

namespace profile {
namespace profile04 {

Checker::Checker( Config cfg ) : config{std::move( cfg )}, counter{0xFFFFU}, checkMutex{} {}

/** \uptrace{SWS_CM_01041} */
void Checker::Check( const crc::Buffer&            buffer,
                     E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) noexcept {
    if ( buffer.empty() ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kNoNewData;
    } else if ( !Profile04::IsBufferLengthValid( config, buffer ) ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kError;
    } else {
        uint16_t                    receivedCounter = ReadCounter( buffer );
        std::lock_guard<std::mutex> lock{checkMutex};
        ProfileCheckStatus          checkStatus =
            DoChecks( static_cast<uint16_t>( buffer.size() ), ReadLength( buffer ), counter,
                      receivedCounter, config.dataId, ReaddataId( buffer ),
                      Profile04::ComputeCrc( config, buffer ), ReadCrc( buffer ) );
        if ( checkStatus != ProfileCheckStatus::kError ) {
            counter = receivedCounter;
        }

        genericProfileCheckStatus = MapProfileCheckStatusToCheckStatus( checkStatus );
    }
}

/** \uptrace{SWS_CM_01041} */
uint16_t Checker::ReadLength( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint16_t>( buffer[ config.offset ] ) << 8U ) |
           ( static_cast<uint16_t>( buffer[ config.offset + 1U ] ) );
}

/** \uptrace{SWS_CM_01041} */
uint16_t Checker::ReadCounter( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint16_t>( buffer[ config.offset + 2U ] ) << 8U ) |
           ( static_cast<uint16_t>( buffer[ config.offset + 3U ] ) );
}

/** \uptrace{SWS_CM_01041} */
uint32_t Checker::ReaddataId( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint32_t>( buffer[ config.offset + 4U ] ) << 24U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 5U ] ) << 16U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 6U ] ) << 8U ) |
           static_cast<uint32_t>( buffer[ config.offset + 7U ] );
}

/** \uptrace{SWS_CM_01041} */
uint32_t Checker::ReadCrc( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint32_t>( buffer[ config.offset + 8U ] ) << 24U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 9U ] ) << 16U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 10U ] ) << 8U ) |
           static_cast<uint32_t>( buffer[ config.offset + 11U ] );
}

/** \uptrace{SWS_CM_01041} */
ProfileCheckStatus Checker::DoChecks( uint16_t length, uint16_t receivedLength,
                                      uint16_t currentCounter, uint16_t receivedCounter,
                                      uint32_t dataId, uint32_t receivedDataId,
                                      uint32_t computedCRC, uint32_t receivedCRC ) noexcept {
    if ( ( computedCRC == receivedCRC ) && ( dataId == receivedDataId ) &&
         ( length == receivedLength ) ) {
        return CheckCounter( receivedCounter, currentCounter );
    }
    return ProfileCheckStatus::kError;
}

ProfileCheckStatus Checker::CheckCounter( uint16_t receivedCounter,
                                          uint16_t currentCounter ) noexcept {
    const uint16_t delta =
        static_cast<uint16_t>( receivedCounter - currentCounter );  // Wrapping around is fine.
    if ( delta > config.maxDeltaCounter ) {
        return ProfileCheckStatus::kWrongSequence;
    } else {
        switch ( delta ) {
            case 0:
                return ProfileCheckStatus::kRepeated;
            case 1:
                return ProfileCheckStatus::kOk;
            default:
                return ProfileCheckStatus::kOkSomeLost;
        }
    }
}
}  // namespace profile04
}  // namespace profile
/* EOF */
