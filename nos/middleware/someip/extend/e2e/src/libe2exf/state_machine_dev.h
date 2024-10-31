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
#ifndef INCLUDE_COM_LIBE2EXF_STATE_MACHINE_DEV_H
#define INCLUDE_COM_LIBE2EXF_STATE_MACHINE_DEV_H

#include "ne_someip_e2e_state_machine.h"

namespace e2exf {

std::ostream& operator<<(
    std::ostream&                                      os,
    const E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus );
std::ostream& operator<<( std::ostream& os, const E2E_state_machine::E2EState& smState );

}  // namespace e2exf

#endif  // INCLUDE_COM_LIBE2EXF_STATE_MACHINE_DEV_H
