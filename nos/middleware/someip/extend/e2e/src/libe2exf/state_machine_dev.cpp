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
#include "state_machine_dev.h"

namespace e2exf {

std::ostream& operator<<(
    std::ostream&                                      os,
    const E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) {
    switch ( genericProfileCheckStatus ) {
        case E2E_state_machine::E2ECheckStatus::kOk:
            os << "OK";
            break;
        case E2E_state_machine::E2ECheckStatus::kRepeated:
            os << "Repeated";
            break;
        case E2E_state_machine::E2ECheckStatus::kWrongSequence:
            os << "WrongSequence";
            break;
        case E2E_state_machine::E2ECheckStatus::kError:
            os << "Error";
            break;
        case E2E_state_machine::E2ECheckStatus::kNotAvailable:
            os << "NOT_AVAILABLE";
            break;
        case E2E_state_machine::E2ECheckStatus::kNoNewData:
            os << "NoNewData";
            break;
        default:
            os << "==UNKNOWN==";
            break;
    }
    return os;
}

std::ostream& operator<<( std::ostream& os, const E2E_state_machine::E2EState& smState ) {
    switch ( smState ) {
        case E2E_state_machine::E2EState::kInit:
            os << "INIT";
            break;
        case E2E_state_machine::E2EState::kInvalid:
            os << "INVALID";
            break;
        case E2E_state_machine::E2EState::kNoData:
            os << "NODATA";
            break;
        case E2E_state_machine::E2EState::kValid:
            os << "VALID";
            break;
        default:
            os << "==UNKNOWN==";
            break;
    }
    return os;
}

}  // namespace e2exf
