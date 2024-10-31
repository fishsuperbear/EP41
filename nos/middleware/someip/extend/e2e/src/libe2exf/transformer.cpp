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
#include <iostream>
#include <utility>
#include "ne_someip_log.h"
#include "extend/e2exf/transformer.h"
#include "config_reader.h"
#include "fault_injector.h"

namespace e2exf {

const e2e::Result Transformer::correctResult{E2E_state_machine::E2EState::kInit,
                                             E2E_state_machine::E2ECheckStatus::kOk};

Transformer::Transformer() : config{} { ne_someip_log_debug("vE2E Transformer not used"); }

Transformer::Transformer( Config&& cfg ) : config{std::move( cfg )} {}

Transformer::Transformer( const std::string& filePath,
                          ConfigurationFormat      configurationFormat )
    : config( LoadE2EConfiguration( filePath, configurationFormat, retLoadE2EConfig ) ) {}

void
Transformer::Protect( const DataIdentifier id, crc::Buffer& buffer ) {
    const auto& protectorIt = config.profileProtectors.find( id );
    if ( protectorIt == config.profileProtectors.end() ) {
        throw std::invalid_argument{"Profile protector does not exist"};
    }

    auto     protector = protectorIt->second;
    uint32_t profileHeaderLength{protector->GetHeaderLength()};
    uint32_t someipHeaderLength{protector->GetHeaderOffset()};
    if ( buffer.size() < ( someipHeaderLength + profileHeaderLength ) ) {
        throw std::out_of_range{"StdReturn::E_SAFETY_HARD_RUNTIMEERROR"};
    }
    protector->Protect( buffer );
}

e2e::Result
Transformer::Check( const DataIdentifier id, const crc::Buffer& buffer ) {
    const auto& checkerIt      = config.profileCheckers.find( id );
    const auto& stateMachineIt = config.stateMachines.find( id );
    if ( checkerIt == config.profileCheckers.end() ||
         stateMachineIt == config.stateMachines.end() ) {
        throw std::invalid_argument{"Data identifier is invalid"};
    }

    const auto& checker = checkerIt->second;
    uint32_t    profileHeaderLength{checker->GetHeaderLength()};
    uint32_t    someipHeaderLength{checker->GetHeaderOffset()};
    if ( buffer.size() < ( someipHeaderLength + profileHeaderLength ) ) {
        throw std::out_of_range{"Data buffer is invalid"};
    }

    /* TODO: P01P02 [SWS_CM_90421] */
    E2E_state_machine::E2ECheckStatus genericProfileCheckStatus;
    E2E_state_machine::E2EState       smState;
    checker->Check( buffer, genericProfileCheckStatus );

    /* TODO: P01P02 [SWS_CM_90421] */
    const auto& stateMachine = stateMachineIt->second;
    stateMachine->Check( genericProfileCheckStatus, smState );

    return e2e::Result{smState, genericProfileCheckStatus};
}

void
Transformer::ProtectOutOfPlace( const DataIdentifier         id,
                                const crc::Buffer& inputBuffer,
                                crc::Buffer&       outputBuffer ) {
    const auto& protectorIt = config.profileProtectors.find( id );
    if ( protectorIt != config.profileProtectors.end() ) {
#if defined( E2E_DEBUG )
        ne_someip_log_debug("Sender: Transformer::protect.");
#endif
        const auto& protector = protectorIt->second;
        uint32_t    profileHeaderOffset{protector->GetHeaderOffset()};
        if ( inputBuffer.size() < profileHeaderOffset ) {
            throw std::out_of_range{"StdReturn::E_SAFETY_HARD_RUNTIMEERROR"};
        }

        uint32_t profileHeaderLength{protector->GetHeaderLength()};
        outputBuffer.resize( inputBuffer.size() + profileHeaderLength );
        std::copy( inputBuffer.begin(), inputBuffer.begin() + profileHeaderOffset,
                   outputBuffer.begin() );
        std::copy( inputBuffer.begin() + profileHeaderOffset, inputBuffer.end(),
                   outputBuffer.begin() + profileHeaderOffset + profileHeaderLength );
        protector->Protect( outputBuffer );
    } else {
#if defined( E2E_DEBUG )
        ne_someip_log_debug("Sender: Transformer: no protection.");
#endif
        outputBuffer = inputBuffer;
    }
}

e2e::Result
Transformer::CheckOutOfPlace( const DataIdentifier         id,
                              const crc::Buffer& inputBuffer,
                              crc::Buffer&       outputBuffer ) {
    const auto& checkerIt      = config.profileCheckers.find( id );
    const auto& stateMachineIt = config.stateMachines.find( id );
    if ( ( checkerIt == config.profileCheckers.end() ) ||
         stateMachineIt == config.stateMachines.end() ) {
        ne_someip_log_debug("Receiver: Transformer: no protection.");
        outputBuffer = inputBuffer;
        return Transformer::correctResult;
    }

    const auto& checker = checkerIt->second;
    uint32_t    profileHeaderLength{checker->GetHeaderLength()};
    uint32_t    someipHeaderLength{checker->GetHeaderOffset()};
    std::size_t inputBufferLength = inputBuffer.size();

#if defined( E2E_DEVELOPMENT )
    lastValidBuffer.resize( inputBufferLength - profileHeaderLength );
    std::copy( inputBuffer.begin(), inputBuffer.begin() + someipHeaderLength,
               lastValidBuffer.begin() );
    std::copy( inputBuffer.begin() + someipHeaderLength + profileHeaderLength, inputBuffer.end(),
               lastValidBuffer.begin() + someipHeaderLength );
    if ( !FaultInjector::Corrupt( inputBuffer ) ) {
        return Result{E2E_state_machine::E2EState::Invalid,
                      E2E_state_machine::E2ECheckStatus::NoNewData};
    }
#endif

#if defined( E2E_DEBUG )
    ne_someip_log_debug("\nReceiver: Transformer::check.");
#endif

    if ( inputBufferLength < ( someipHeaderLength + profileHeaderLength ) ) {
        throw std::out_of_range{"StdReturn::E_SAFETY_HARD_RUNTIMEERROR"};
    }
    /* TODO: P01P02 [SWS_CM_90421] */
    E2E_state_machine::E2ECheckStatus profileProfileCheckStatus;
    E2E_state_machine::E2EState       smState;
    checker->Check( inputBuffer, profileProfileCheckStatus );

    const auto& stateMachine = stateMachineIt->second;
    stateMachine->Check( profileProfileCheckStatus, smState );

    /* TODO: P01P02 [SWS_CM_90421] */
    outputBuffer.resize( inputBufferLength - profileHeaderLength );
    std::copy( inputBuffer.begin(), inputBuffer.begin() + someipHeaderLength,
               outputBuffer.begin() );
    std::copy( inputBuffer.begin() + someipHeaderLength + profileHeaderLength, inputBuffer.end(),
               outputBuffer.begin() + someipHeaderLength );
    return e2e::Result{smState, profileProfileCheckStatus};
}

bool
Transformer::IsProtected( const DataIdentifier id ) const {
    return ( ( config.profileCheckers.find( id ) != config.profileCheckers.end() ) ||
             ( config.stateMachines.find( id ) != config.stateMachines.end() ) );
}

bool
Transformer::GetCounter(const DataIdentifier id, const crc::Buffer& buffer, std::uint32_t& counter) {
    const auto& checkerIt      = config.profileCheckers.find( id );
    if ( checkerIt == config.profileCheckers.end() ) {
        ne_someip_log_debug("Receiver: Transformer: no protection.");
        return false;
    }
    const auto& checker = checkerIt->second;
    counter = checker->GetCounter(buffer);

    return true;
}

}  // namespace e2exf
