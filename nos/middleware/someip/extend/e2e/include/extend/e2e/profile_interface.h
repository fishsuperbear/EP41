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

#ifndef E2E_INCLUDE_E2E_PROFILE_INTERFACE_H_
#define E2E_INCLUDE_E2E_PROFILE_INTERFACE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>

namespace profile {

/// @brief Supported E2E profile types
enum class ProfileName : uint8_t {
    PROFILE_01,
    PROFILE_02,
    PROFILE_04,
    PROFILE_05,
    PROFILE_06,
    PROFILE_07,
    PROFILE_11,
    PROFILE_22,
    PROFILE_UNKNOW
};

namespace profile_interface {

/// @brief Generic Interface of a profile
class ProfileInterface {
   public:
    /// @brief Default destructor
    virtual ~ProfileInterface() noexcept = default;
    /// @brief Returns offset of the E2E header
    ///
    /// @return offset of E2E header
    virtual uint32_t GetHeaderOffset() const noexcept = 0;
    /// @brief Returns E2E header length
    ///
    /// @return length of E2E header
    virtual uint32_t GetHeaderLength() const noexcept = 0;
};

}  // namespace profile_interface
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_PROFILE_INTERFACE_H_
/* EOF */
