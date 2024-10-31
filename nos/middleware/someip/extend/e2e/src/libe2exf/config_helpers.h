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
#ifndef INCLUDE_COM_LIBE2EXF_CONFIG_HELPERS_H
#define INCLUDE_COM_LIBE2EXF_CONFIG_HELPERS_H

#include <memory>
#include "extend/e2exf/config.h"
#include "extend/e2exf/types.h"

namespace e2exf {

template <typename ProfileType, typename ProfileConfigType>
void RegisterProfileProtector( Config& transformerConfig, const DataIdentifier dataIdentifier,
                               const ProfileConfigType& profileConfig ) {
    if ( transformerConfig.profileProtectors.find( dataIdentifier ) !=
         transformerConfig.profileProtectors.end() ) {
        throw std::invalid_argument{"Wrong configuration of E2E: overlapping configuration"};
    }
    transformerConfig.profileProtectors[ dataIdentifier ] =
        std::make_shared<ProfileType>( profileConfig );
}

template <typename ProfileType, typename ProfileConfigType>
void RegisterProfileChecker( Config& transformerConfig, const DataIdentifier dataIdentifier,
                             const ProfileConfigType& profileConfig ) {
    if ( transformerConfig.profileCheckers.find( dataIdentifier ) !=
         transformerConfig.profileCheckers.end() ) {
        throw std::invalid_argument{"Wrong configuration of E2E: overlapping configuration"};
    }
    transformerConfig.profileCheckers[ dataIdentifier ] =
        std::make_shared<ProfileType>( profileConfig );
}

template <typename ProtectorType, typename CheckerType, typename ProfileConfigType>
void RegisterProfile( Config& config, const DataIdentifier dataIdentifier,
                      const ProfileConfigType& profileConfig, bool isProtector, bool isChecker ) {
    if ( isProtector ) {
        RegisterProfileProtector<ProtectorType>( config, dataIdentifier, profileConfig );
    }
    if ( isChecker ) {
        RegisterProfileChecker<CheckerType>( config, dataIdentifier, profileConfig );
    }
}

inline void RegisterStateMachine( Config& transformerConfig, const DataIdentifier dataIdentifier,
                                  const E2E_state_machine::Config& stateMachineConfig ) {
    if ( transformerConfig.stateMachines.find( dataIdentifier ) !=
         transformerConfig.stateMachines.end() ) {
        throw std::invalid_argument{"Wrong configuration of E2E: overlapping configuration"};
    }

    transformerConfig.stateMachines[ dataIdentifier ] =
        std::make_shared<E2E_state_machine::StateMachine>( stateMachineConfig );
}

}  // namespace e2exf

#endif  // INCLUDE_COM_LIBE2EXF_CONFIG_HELPERS_H
/* EOF */
