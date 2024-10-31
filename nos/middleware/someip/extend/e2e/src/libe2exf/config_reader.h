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
#ifndef INCLUDE_COM_LIBE2EXF_CONFIG_READER_H
#define INCLUDE_COM_LIBE2EXF_CONFIG_READER_H

#include <string>
#include <cstdint>
#include <tuple>
#include "extend/e2exf/config.h"
#include "extend/e2exf/types.h"

namespace e2exf {

Config LoadE2EConfiguration( const std::string& filePath, ConfigurationFormat format, bool& ret );

std::map<std::tuple<std::uint16_t, std::uint16_t, std::uint16_t>, DataIdentifier>
    LoadE2EDataIdMapping( const std::string& filePath, ConfigurationFormat format, bool& ret );

std::map<e2exf::DataIdentifier,
               std::shared_ptr<E2E_state_machine::StateMachine>>
    LoadE2EStateMachines( const std::string& filePath, ConfigurationFormat format, bool& ret );

}  // namespace e2exf

#endif  // INCLUDE_COM_LIBE2EXF_CONFIG_READER_H
