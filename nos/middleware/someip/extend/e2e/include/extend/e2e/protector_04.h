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
#ifndef E2E_INCLUDE_E2E_PROTECTOR04_H_
#define E2E_INCLUDE_E2E_PROTECTOR04_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <mutex>
#include "extend/e2e/profile_04.h"
#include "extend/e2e/protector_interface.h"

namespace profile {
namespace profile04 {

/// @brief Implementation of Protector Interface for E2E profile 04
class Protector final : public profile::profile_interface::ProtectorInterface {
   public:
    Protector() = delete;
    /// @brief Creates Protector instance using given profile configuration
    ///
    /// @param cfg Profile configuration
    ///
    /// @uptrace{SWS_CM_90433}
    explicit Protector( Config cfg );
    ~Protector() noexcept override = default;

    Protector( const Protector& oth )     = delete;
    Protector( Protector&& oth ) noexcept = delete;
    Protector& operator=( const Protector& oth ) = delete;
    Protector& operator=( Protector&& oth ) noexcept = delete;

    void     Protect( crc::Buffer& buffer ) override;
    uint32_t GetHeaderLength() const noexcept override { return Profile04::headerLength; }
    uint32_t GetHeaderOffset() const noexcept override { return config.offset; }

   private:
    void WriteLength( crc::Buffer& buffer, uint16_t length ) noexcept;
    void WriteCounter( crc::Buffer& buffer, uint16_t currentCounter ) noexcept;
    void WritedataId( crc::Buffer& buffer, uint32_t dataId ) noexcept;
    void WriteCrc( crc::Buffer& buffer, uint32_t computedCRC ) noexcept;
    void IncrementCounter() noexcept;

   private:
    const Config config;
    uint16_t     counter;
    std::mutex   protectMutex;
};

}  // namespace profile04
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_PROTECTOR04_H_
/* EOF */
