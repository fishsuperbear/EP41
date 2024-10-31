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
#ifndef E2E_INCLUDE_E2E_CHECKER_INTERFACE_H_
#define E2E_INCLUDE_E2E_CHECKER_INTERFACE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "extend/crc/buffer.h"
#include "extend/e2e/profile_interface.h"
#include "ne_someip_e2e_state_machine.h"

namespace profile {

/// @brief Profile-independent status of the reception on one single Data in one cycle.
/// @note This enumeration left for compatibility with classic platform
///
/// @uptrace{SWS_E2E_00347}
/// @uptrace{SWS_CM_90421}
enum class ProfileCheckStatus : uint8_t {
    kOk,             ///< OK: The checks of the Data in this cycle were successful
    kRepeated,       ///< Error: the checks of the Data in this cycle were successful, with the exception of the repetition
    kWrongSequence,  ///< Error: the checks of the Data in this cycle were successful, with the exception of counter jump, which changed more than the allowed delta
    kError,          ///< Error: error not related to counters occurred (e.g. wrong CRC, wrong length, wrong options, wrong Data ID)
    kNotAvailable,   ///< No value has been received yet (e.g. during initialization). This is used as the initialization value for the buffer.
    kNoNewData,      ///< Error: The Check function has been invoked but no new Data is not available since the last call, according to communication medium
    kCheckDisabled,  ///< No E2E check status available (no E2E protection is configured).
    kOkSomeLost,     ///< OK: the checks of the Data in this cycle were successful (including counter check, which was incremented within the allowed configured delta)
};

/// @brief Converts ProfileCheckStatus type  to CheckStatus type of StateMachine
///
/// @param status profile check status
///
/// @return StateMachine CheckStatus type
inline E2E_state_machine::E2ECheckStatus MapProfileCheckStatusToCheckStatus(
    ProfileCheckStatus status ) {
    switch ( status ) {
        case ProfileCheckStatus::kOk:
            return E2E_state_machine::E2ECheckStatus::kOk;
        case ProfileCheckStatus::kOkSomeLost:
            return E2E_state_machine::E2ECheckStatus::kOk;
        case ProfileCheckStatus::kRepeated:
            return E2E_state_machine::E2ECheckStatus::kRepeated;
        case ProfileCheckStatus::kWrongSequence:
            return E2E_state_machine::E2ECheckStatus::kWrongSequence;
        case ProfileCheckStatus::kError:
            return E2E_state_machine::E2ECheckStatus::kError;
        case ProfileCheckStatus::kNotAvailable:
            return E2E_state_machine::E2ECheckStatus::kNotAvailable;
        case ProfileCheckStatus::kNoNewData:
            return E2E_state_machine::E2ECheckStatus::kNoNewData;
        case ProfileCheckStatus::kCheckDisabled:
            return E2E_state_machine::E2ECheckStatus::kCheckDisabled;
    };
    // functions that use this were declared noexcept
    return E2E_state_machine::E2ECheckStatus::kError;
}

namespace profile_interface {

/// @brief Interface of E2E checker.
///
/// E2E performs check routine on given data.
class CheckerInterface : public ProfileInterface {
   public:
    /// @brief Default destructor
    ~CheckerInterface() noexcept override = default;

    /// @brief Check routine performs validation of E2E header that is supplied within other data in
    /// buffer parameter
    ///
    /// @param buffer                     buffer with payload and E2E header
    /// @param genericProfileCheckStatus  Result of check routine
    virtual void Check(
        const crc::Buffer&            buffer,
        E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) noexcept = 0;

    virtual uint32_t GetCounter( const crc::Buffer& buffer ) const noexcept = 0;
};

}  // namespace profile_interface
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_CHECKER_INTERFACE_H_
