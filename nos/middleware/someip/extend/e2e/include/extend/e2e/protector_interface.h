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
#ifndef E2E_INCLUDE_E2E_PROTECTOR_INTERFACE_H_
#define E2E_INCLUDE_E2E_PROTECTOR_INTERFACE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "extend/crc/buffer.h"
#include "extend/e2e/profile_interface.h"

namespace profile {
namespace profile_interface {

/// @brief Interface of E2E protector
class ProtectorInterface : public ProfileInterface {
   public:
    /// @brief Default destructor
    ~ProtectorInterface() noexcept override = default;

    /// @brief Protect routine embed E2E header to given buffer
    ///
    /// @param buffer data to be protected
    virtual void Protect( crc::Buffer& buffer ) noexcept( false ) = 0;
};

}  // namespace profile_interface
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_PROTECTOR_INTERFACE_H_
/* EOF */
