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
#ifndef E2E_INCLUDE_E2EXF_CONFIG_H_
#define E2E_INCLUDE_E2EXF_CONFIG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <memory>
#include <cstdint>
#include <map>
#include "extend/e2e/checker_interface.h"
#include "extend/e2e/protector_interface.h"
#include "ne_someip_e2e_state_machine.h"
#include "extend/e2exf/types.h"

namespace e2exf {

/// @brief Configuration of E2E data transformer
struct Config {
    /// @brief profile protectors associated with data identifiers
    std::map<DataIdentifier,
                   std::shared_ptr<profile::profile_interface::ProtectorInterface>>
        profileProtectors;
    /// @brief profile checkers associated with data identifiers
    std::map<DataIdentifier,
                   std::shared_ptr<profile::profile_interface::CheckerInterface>>
        profileCheckers;
    /// @brief E2E state machines associated with data identifiers
    std::map<DataIdentifier, std::shared_ptr<E2E_state_machine::StateMachine>>
        stateMachines;
};

}  // namespace e2exf

#endif  // E2E_INCLUDE_E2EXF_CONFIG_H_
/* EOF */
