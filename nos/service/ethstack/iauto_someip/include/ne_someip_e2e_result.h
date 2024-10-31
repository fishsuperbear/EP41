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

#ifndef E2E_INCLUDE_E2E_RESULT_H_
#define E2E_INCLUDE_E2E_RESULT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "ne_someip_e2e_state_machine.h"

namespace e2e {
/// @brief   Result of E2E check - a combination of E2E StateMachine state and E2E header check
/// status
///
/// @uptrace{SWS_CM_90423}
class Result {
   public:
    /// @brief E2E StateMachine state
    using E2EState = e2e::SMState;
    /// @brief E2E profile check status
    using E2ECheckStatus = e2e::ProfileCheckStatus;

    /// @brief Constructs default result with StateMachine state set to NoData and check status -
    /// NotAvailable
    Result() noexcept;

    /// @brief Constructs result of E2E check
    ///
    /// @param state        E2E StateMachine state
    /// @param checkStatus  status of E2E check
    Result( E2EState state, E2ECheckStatus checkStatus ) noexcept;

    /// @brief Default copy constructor
    Result( const Result& ) = default;

    /// @brief Default move constructor
    Result( Result&& ) noexcept = default;

    /// @brief Default assignemnt operator
    Result& operator=( const Result& ) = default;

    /// @brief Default move-assignment operator
    Result& operator=( Result&& ) noexcept = default;

    /// @brief Default destructor
    ~Result() noexcept = default;

    /// @brief Checks if result of E2E check is valid
    ///
    /// @return true if state is "Valid" and check status is "Ok"
    bool IsOK() const noexcept;

    /// @brief accessor to StateMachine state
    ///
    /// @return state of E2E StateMachine
    E2EState GetSMState() const noexcept { return smState; }

    /// @brief accessor to check status
    ///
    /// @return check status
    ///
    /// @uptrace{SWS_CM_90420}
    E2ECheckStatus GetProfileCheckStatus() const noexcept { return checkStatus; }

   private:
    E2EState       smState;
    E2ECheckStatus checkStatus;
};
}  // namespace e2e
#endif  // E2E_INCLUDE_E2E_RESULT_H_
/* EOF */
