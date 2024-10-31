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
#include "extend/e2e/checker_11.h"
#include <utility>

namespace profile {
namespace profile11 {

Checker::Checker( Config cfg )
    : config{std::move( cfg )}, counter{Profile11::maxCounterValue}, checkMutex{} {}

void Checker::Check( const crc::Buffer&            buffer,
                     E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) noexcept {
    if ( buffer.empty() ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kNoNewData;
    } else if ( !Profile11::IsBufferLengthValid( config, buffer ) ) {
        genericProfileCheckStatus = E2E_state_machine::E2ECheckStatus::kError;
    } else {
        uint8_t                     receivedCounter{ReadCounter( buffer )};
        std::lock_guard<std::mutex> lock( checkMutex );
        ProfileCheckStatus          checkStatus =
            DoChecks( config.dataId, ReadDataIdNibble( buffer ), counter, receivedCounter,
                      Profile11::ComputeCrc( config, buffer ), ReadCrc( buffer ) );
        if ( checkStatus != ProfileCheckStatus::kError ) {
            counter = receivedCounter;
        }

        genericProfileCheckStatus = MapProfileCheckStatusToCheckStatus( checkStatus );
    }
}

uint8_t Checker::ReadDataIdNibble( const crc::Buffer& buffer ) const noexcept {
    uint8_t dataIdNibble{0U};
    if ( config.dataIdMode == DataIdMode::LOWER_12_BIT ) {
        dataIdNibble = ( ( buffer[ config.crcOffset + 1U ] & 0xF0U ) >> 4U );
    }
    return dataIdNibble;
}

uint8_t Checker::ReadCounter( const crc::Buffer& buffer ) const noexcept {
    return ( buffer[ config.crcOffset + 1U ] & 0x0FU );
}

uint8_t Checker::ReadCrc( const crc::Buffer& buffer ) const noexcept {
    return ( buffer[ config.crcOffset ] );
}

ProfileCheckStatus Checker::DoChecks( uint16_t dataId, uint8_t receivedDataIdNibble,
                                      uint8_t currentCounter, uint8_t receivedCounter,
                                      uint8_t computedCrc, uint8_t receivedCrc ) const {
    bool areCrcEqual = ( receivedCrc == computedCrc );
    bool isDataIdModeValid =
        ( ( config.dataIdMode == DataIdMode::ALL_16_BIT ) ||
          ( config.dataIdMode == DataIdMode::LOWER_12_BIT &&
            receivedDataIdNibble == Profile11::CalculateDataIdNibble( dataId ) ) );
    if ( areCrcEqual && isDataIdModeValid ) {
        uint8_t delta{CalculateCounterDelta( receivedCounter, currentCounter )};
        if ( delta <= config.maxDeltaCounter )  // && delta >= 0)
        {
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
    return ProfileCheckStatus::kError;
}

uint8_t Checker::CalculateCounterDelta( uint8_t receivedCounter, uint8_t currentCounter ) const
    noexcept {
    if ( receivedCounter == 0x0U && currentCounter == Profile11::maxCounterValue ) {
        return 1U;
    }
    return ( receivedCounter - currentCounter );
}

}  // namespace profile11
}  // namespace profile
/* EOF */
