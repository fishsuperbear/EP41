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
#include "extend/e2e/checker_07.h"
#include <utility>

namespace profile {
namespace profile07 {

Checker::Checker( Config cfg ) : config{std::move( cfg )}, counter{0xFFFFFFFFU}, checkMutex{} {}

/** \uptrace{SWS_CM_01041} */
void Checker::Check( const crc::Buffer&            buffer,
                     E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) noexcept {
    if ( buffer.empty() ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kNoNewData;
    } else if ( !Profile07::IsBufferLengthValid( config, buffer ) ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kError;
    } else {
        uint32_t                    receivedCounter = ReadCounter( buffer );
        std::lock_guard<std::mutex> lock{checkMutex};
        ProfileCheckStatus          checkStatus = DoChecks(
            buffer.size(), ReadLength( buffer ), counter, receivedCounter, config.dataId,
            ReadDataId( buffer ), Profile07::ComputeCrc( config, buffer ), ReadCrc( buffer ) );
        if ( checkStatus != ProfileCheckStatus::kError ) {
            counter = receivedCounter;
        }

        genericProfileCheckStatus = MapProfileCheckStatusToCheckStatus( checkStatus );
    }
}

/**  \uptrace{SWS_CM_01041} */
uint32_t Checker::ReadLength( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint32_t>( buffer[ config.offset + 8U ] ) << 24U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 9U ] ) << 16U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 10U ] ) << 8U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 11U ] ) );
}

/** \uptrace{SWS_CM_01041} */
uint32_t Checker::ReadCounter( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint32_t>( buffer[ config.offset + 12U ] ) << 24U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 13U ] ) << 16U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 14U ] ) << 8U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 15U ] ) );
}

/** \uptrace{SWS_CM_01041} */
uint32_t Checker::ReadDataId( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint32_t>( buffer[ config.offset + 16U ] ) << 24U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 17U ] ) << 16U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 18U ] ) << 8U ) |
           ( static_cast<uint32_t>( buffer[ config.offset + 19U ] ) );
}

/** \uptrace{SWS_CM_01041} */
uint64_t Checker::ReadCrc( const crc::Buffer& buffer ) const noexcept {
    return ( static_cast<uint64_t>( buffer[ config.offset ] ) << 56U ) |
           ( static_cast<uint64_t>( buffer[ config.offset + 1U ] ) << 48U ) |
           ( static_cast<uint64_t>( buffer[ config.offset + 2U ] ) << 40U ) |
           ( static_cast<uint64_t>( buffer[ config.offset + 3U ] ) << 32U ) |
           ( static_cast<uint64_t>( buffer[ config.offset + 4U ] ) << 24U ) |
           ( static_cast<uint64_t>( buffer[ config.offset + 5U ] ) << 16U ) |
           ( static_cast<uint64_t>( buffer[ config.offset + 6U ] ) << 8U ) |
           ( static_cast<uint64_t>( buffer[ config.offset + 7U ] ) );
}

/** \uptrace{SWS_CM_01041} */
ProfileCheckStatus Checker::DoChecks( uint32_t length, uint32_t receivedLength,
                                      uint32_t currentCounter, uint32_t receivedCounter,
                                      uint32_t dataId, uint32_t receivedDataId,
                                      uint64_t computedCRC, uint64_t receivedCRC ) noexcept {
    if ( ( computedCRC == receivedCRC ) && ( dataId == receivedDataId ) &&
         ( length == receivedLength ) ) {
        return CheckCounter( receivedCounter, currentCounter );
    }
    return ProfileCheckStatus::kError;
}

ProfileCheckStatus Checker::CheckCounter( uint32_t receivedCounter,
                                          uint32_t currentCounter ) noexcept {
    const uint32_t delta{receivedCounter - currentCounter};
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
}  // namespace profile07
}  // namespace profile
/* EOF */
