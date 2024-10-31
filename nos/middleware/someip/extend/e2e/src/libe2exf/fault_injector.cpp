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
#include <ctime>
#include <iostream>
#include "fault_injector.h"

namespace e2exf {

FaultInjector::FaultMethod FaultInjector::pass =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& history ) {
        (void) inputBuffer;
        (void) history;
        return true;
    };
FaultInjector::FaultMethod FaultInjector::lost =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& history ) {
        (void) inputBuffer;
        (void) history;
        return false;
    };

FaultInjector::FaultMethod FaultInjector::repeat =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& history ) {
        inputBuffer = history.back();
        return true;
    };
FaultInjector::FaultMethod FaultInjector::singleBitCorruption =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& ) {
        std::srand( std::time( nullptr ) );
        uint8_t  corruptedByteMask     = 1U << ( std::rand() % 8U );
        uint32_t corruptedBytePosition = std::rand() % inputBuffer.size();
        inputBuffer[ corruptedBytePosition ] ^= corruptedByteMask;
        return true;
    };
FaultInjector::FaultMethod FaultInjector::singleBitCorruptionInE2EHeader =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& ) {
        std::srand( std::time( nullptr ) );
        uint8_t  corruptedByteMask     = 1U << ( std::rand() % 8U );
        uint32_t corruptedBytePosition = std::rand() % 12U;
        inputBuffer[ 16U + corruptedBytePosition ] ^= corruptedByteMask;
        return true;
    };
FaultInjector::FaultMethod FaultInjector::partialDelivery =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& ) {
        // trim last byte
        inputBuffer.resize( inputBuffer.size() - 1U );
        return true;
    };
FaultInjector::FaultMethod FaultInjector::swapBytes =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& ) {
        std::srand( std::time( nullptr ) );
        uint32_t firstBytePosition  = std::rand() % inputBuffer.size();
        uint32_t secondBytePosition = std::rand() % inputBuffer.size();
        while ( ( firstBytePosition == secondBytePosition ) ||
                ( inputBuffer[ firstBytePosition ] == inputBuffer[ secondBytePosition ] ) ) {
            secondBytePosition = std::rand() % inputBuffer.size();
        }
        std::swap( inputBuffer[ firstBytePosition ], inputBuffer[ secondBytePosition ] );
        return true;
    };
FaultInjector::FaultMethod FaultInjector::repeatTwoStepsBefore =
    []( crc::Buffer& inputBuffer, std::vector<crc::Buffer>& history ) {
        inputBuffer = history[ history.size() - 2U ];
        return true;
    };
FaultInjector::FaultMethod FaultInjector::Resend( uint32_t historyQueueOffset ) {
    return [historyQueueOffset]( crc::Buffer&                    inputBuffer,
                                 std::vector<crc::Buffer>& history ) {
        // trim last byte
        inputBuffer = history[ history.size() - historyQueueOffset ];
        return true;
    };
}

std::vector<crc::Buffer>      FaultInjector::history;
std::queue<FaultInjector::FaultMethodWithLog> FaultInjector::faultMethodQueue;

void FaultInjector::RegisterMethod( FaultInjector::FaultMethodWithLog methodWithLog ) {
    faultMethodQueue.push( methodWithLog );
}
bool FaultInjector::Corrupt( crc::Buffer& inputBuffer ) {
    bool ret = true;
    if ( !faultMethodQueue.empty() ) {
        FaultMethodWithLog faultMethodWithLog = faultMethodQueue.front();
        ret                                   = faultMethodWithLog.first( inputBuffer, history );
        if ( faultMethodWithLog.second != "" ) {
            std::cout << faultMethodWithLog.second << std::endl;
        }
        history.push_back( inputBuffer );
        faultMethodQueue.pop();
    }
    return ret;
}
}  // namespace e2exf
