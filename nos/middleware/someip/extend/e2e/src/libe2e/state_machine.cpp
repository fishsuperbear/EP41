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
#include "ne_someip_e2e_state_machine.h"

namespace E2E_state_machine {

StateMachine::StateMachine( const E2E_state_machine::Config& cfg )
    : config{cfg}
    , genericProfileCheckStatusWindow{config.windowSize,
                                      E2E_state_machine::E2ECheckStatus::kNotAvailable}
    , windowTopIndex{0U}
    , okCount{0U}
    , errorCount{0U}
    , state{E2E_state_machine::E2EState::kNoData}
    , stateMachineMutex{} {}

StateMachine::StateMachine( const E2E_state_machine::StateMachine& oth )
    : config{oth.config}
    , genericProfileCheckStatusWindow( oth.genericProfileCheckStatusWindow )
    , windowTopIndex{oth.windowTopIndex}
    , okCount{oth.okCount}
    , errorCount{oth.errorCount}
    , state{oth.state}
    , stateMachineMutex{} {}

StateMachine::StateMachine( StateMachine&& oth ) noexcept
    : config{oth.config}
    , genericProfileCheckStatusWindow( std::move( oth.genericProfileCheckStatusWindow ) )
    , windowTopIndex{oth.windowTopIndex}
    , okCount{oth.okCount}
    , errorCount{oth.errorCount}
    , state{oth.state}
    , stateMachineMutex{} {}

StateMachine& StateMachine::operator=( const E2E_state_machine::StateMachine& oth ) {
    if ( this != &oth ) {
        config                          = oth.config;
        genericProfileCheckStatusWindow = oth.genericProfileCheckStatusWindow;
        windowTopIndex                  = oth.windowTopIndex;
        okCount                         = oth.okCount;
        errorCount                      = oth.errorCount;
        state                           = oth.state;
        // We do not copy the stateMachineMutex
    }
    return *this;
}

StateMachine& StateMachine::operator=( StateMachine&& oth ) noexcept {
    if ( this != &oth ) {
        config                          = oth.config;
        genericProfileCheckStatusWindow = std::move( oth.genericProfileCheckStatusWindow );
        windowTopIndex                  = oth.windowTopIndex;
        okCount                         = oth.okCount;
        errorCount                      = oth.errorCount;
        state                           = oth.state;
        // We do not move the stateMachineMutex
    }
    return *this;
}

/** \uptrace{SWS_E2E_00340} */
/** \uptrace{SWS_E2E_00345} */
/**
 * @startuml
 *
 * [*] --> State::NODATA : State::DEINIT / Transition through init()
 *
 * State::NODATA : Check(ProfileStatus, Config, State)
 * State::NODATA --> State::NODATA : [ProfileStatus != ERROR] /
 * AddStatus(ProfileStatus, State)
 * State::NODATA --> State::INIT : [ELSE]
 *
 * State::INIT : Check(ProfileStatus, Config, State) /
 * AddStatus(ProfileStatus, State)
 * State::INIT --> State::VALID : [(State->errorCount <=
 * Config->maxErrorStateInit) && (State->okCount >=
 * Config->minOkStateInit)]
 * State::INIT --> State::INVALID : [(State->errorCount >
 * Config->maxErrorStateInit)]
 * State::INIT --> State::INIT : [ELSE]
 *
 * State::VALID : Check(ProfileStatus, Config, State) /
 * AddStatus(ProfileStatus, State)
 * State::VALID --> State::VALID : [(State->errorCount <=
 * Config->maxErrorStateValid) && (State->okCount
 * >= Config->minOkStateValid)]
 * State::VALID --> State::INVALID : [ELSE]
 *
 * State::INVALID : check(ProfileStatus, Config, State) /
 * addStatus(ProfileStatus, State)
 * State::INVALID --> State::VALID : [(State->errorCount <=
 * Config->MaxErrorStateInwalid) &&
 * (State->okCount >= Config->minOkStateInvalid)]
 * State::INVALID --> State::INVALID : [ELSE]
 *
 * @enduml
 */
void StateMachine::Check( const E2E_state_machine::E2ECheckStatus genericProfileCheckStatus,
                          E2E_state_machine::E2EState&            smState ) noexcept {
    std::lock_guard<std::mutex> lock( stateMachineMutex );
    switch ( state ) {
        case E2EState::kNoData:
            HandleStateNoData( genericProfileCheckStatus );
            break;
        case E2EState::kInit:
            HandleStateInit( genericProfileCheckStatus );
            break;
        case E2EState::kValid:
            HandleStateValid( genericProfileCheckStatus );
            break;
        case E2EState::kInvalid:
            HandleStateInvalid( genericProfileCheckStatus );
            break;
        case E2EState::kStateMDisabled:
            break;
    }
    smState = state;
}

void StateMachine::HandleStateNoData( const E2ECheckStatus profileStatus ) noexcept {
    if ( ( profileStatus != E2ECheckStatus::kError ) &&
         ( profileStatus != E2ECheckStatus::kNoNewData ) ) {
        state = E2EState::kInit;
    } else {
        // Keep the same state
    }
}

void StateMachine::HandleStateInit( const E2ECheckStatus profileStatus ) noexcept {
    AddStatus( profileStatus );
    if ( ( errorCount <= config.maxErrorStateInit ) && ( okCount >= config.minOkStateInit ) ) {
        state = E2EState::kValid;
    } else if ( errorCount > config.maxErrorStateInit ) {
        state = E2EState::kInvalid;
    } else {
        state = E2EState::kInit;
    }
}

void StateMachine::HandleStateValid( const E2ECheckStatus profileStatus ) noexcept {
    AddStatus( profileStatus );
    if ( ( errorCount <= config.maxErrorStateValid ) && ( okCount >= config.minOkStateValid ) ) {
        state = E2EState::kValid;
    } else {
        state = E2EState::kInvalid;
    }
}

void StateMachine::HandleStateInvalid( const E2ECheckStatus profileStatus ) noexcept {
    AddStatus( profileStatus );
    if ( ( errorCount <= config.maxErrorStateInvalid ) &&
         ( okCount >= config.minOkStateInvalid ) ) {
        state = E2EState::kValid;
    } else {
        // Keep the same state
    }
}

/** \uptrace{SWS_E2E_00466} */
void StateMachine::AddStatus( const E2ECheckStatus genericProfileCheckStatus ) noexcept {
    // update counter according to oldest value (overridden in next step)
    switch ( genericProfileCheckStatusWindow[ windowTopIndex ] ) {
        case E2ECheckStatus::kError:
            --errorCount;
            break;
        case E2ECheckStatus::kOk:
            --okCount;
            break;
        default:
            break;
    }

    // update status
    genericProfileCheckStatusWindow[ windowTopIndex ] = genericProfileCheckStatus;

    // set new window top index
    if ( windowTopIndex == ( config.windowSize - 1U ) ) {
        windowTopIndex = 0U;
    } else {
        ++windowTopIndex;
    }

    // update counters according to new value
    switch ( genericProfileCheckStatus ) {
        case E2ECheckStatus::kError:
            ++errorCount;
            break;
        case E2ECheckStatus::kOk:
            ++okCount;
            break;
        default:
            break;
    }
}
}  // namespace E2E_state_machine
/* EOF */
