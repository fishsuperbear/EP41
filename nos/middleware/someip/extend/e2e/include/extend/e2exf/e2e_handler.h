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
#ifndef E2E_INCLUDE_E2EXF_STATUS_HANDLER_H_
#define E2E_INCLUDE_E2EXF_STATUS_HANDLER_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include <string>
#include "extend/crc/buffer.h"
#include "ne_someip_e2e_result.h"
#include "ne_someip_e2e_state_machine.h"
#include "extend/e2exf/types.h"

namespace e2exf {

class E2EHandler {
   public:
    static bool Configure( const std::string&       bindingConfigurationPath,
                           ConfigurationFormat      bindingConfigurationFormat,
                           const std::string&       e2exfConfigurationPath,
                           ConfigurationFormat      e2exfConfigurationFormat );

    static e2e::Result HandleCheckStatus( std::uint16_t serviceId, std::uint16_t instanceId,
                                          std::uint16_t                               eventId,
                                          E2E_state_machine::E2ECheckStatus checkStatus );

    static bool ProtectEvent( const std::uint16_t serviceId, const std::uint16_t instanceId,
                              const std::uint16_t eventId, const crc::Buffer& inputBuffer,
                              crc::Buffer& outputBuffer );

    static e2e::Result CheckEvent( const std::uint16_t serviceId, const std::uint16_t instanceId,
                                   const std::uint16_t eventId, const crc::Buffer& inputBuffer,
                                   crc::Buffer& outputBuffer );

    static bool ProtectMethod( const std::uint16_t serviceId, const std::uint16_t instanceId,
                               const std::uint16_t methodId, const crc::Buffer& inputBuffer,
                               crc::Buffer& outputBuffer );

    static e2e::Result CheckMethod( const std::uint16_t serviceId, const std::uint16_t instanceId,
                                    const std::uint16_t methodId, const crc::Buffer& inputBuffer,
                                    crc::Buffer& outputBuffer );

    static bool IsProtected( const std::uint16_t serviceId, const std::uint16_t instanceId, const std::uint16_t eventOrMethodId );

    static void GetDataIdAndCounter( const std::uint16_t          serviceId,
                                     const std::uint16_t          instanceId,
                                     const std::uint16_t          eventId,
                                     const crc::Buffer& inputBuffer,
                                     DataIdentifier&              dataId,
                                     std::uint32_t&               counter );
};

}  // namespace e2exf

#endif  // E2E_INCLUDE_E2EXF_STATUS_HANDLER_H_
/* EOF */
