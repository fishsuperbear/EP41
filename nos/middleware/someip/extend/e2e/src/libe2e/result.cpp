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
#include "ne_someip_e2e_result.h"

namespace e2e {

Result::Result() noexcept : Result( E2EState::kNoData, E2ECheckStatus::kNotAvailable ) {}

Result::Result( E2EState state, E2ECheckStatus status ) noexcept
    : smState{state}, checkStatus{status} {}

bool Result::IsOK() const noexcept {
    return smState == E2EState::kValid && checkStatus == E2ECheckStatus::kOk;
}

}  // namespace e2e
/* EOF */
