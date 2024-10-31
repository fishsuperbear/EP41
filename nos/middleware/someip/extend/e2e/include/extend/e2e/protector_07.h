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
#ifndef E2E_INCLUDE_E2E_PROTECTOR07_H_
#define E2E_INCLUDE_E2E_PROTECTOR07_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include <mutex>
#include "extend/e2e/profile_07.h"
#include "extend/e2e/protector_interface.h"

namespace profile {
namespace profile07 {

/// @brief Implementation of Protector Interface for E2E profile 07
class Protector final : public profile::profile_interface::ProtectorInterface {
   public:
    Protector() = delete;

    /// @brief Creates Protector instance using given profile configuration
    ///
    /// @param config Profile configuration
    ///
    /// @uptrace{SWS_CM_90433}
    explicit Protector( Config config );
    Protector( const Protector& )     = delete;
    Protector( Protector&& ) noexcept = delete;
    Protector& operator=( const Protector& ) = delete;
    Protector& operator=( Protector&& ) noexcept = delete;
    ~Protector() noexcept override               = default;

    void Protect( crc::Buffer& buffer ) override;

    uint32_t GetHeaderLength() const noexcept override { return Profile07::headerLength; }
    uint32_t GetHeaderOffset() const noexcept override { return config.offset; }

   private:
    void WriteLength( crc::Buffer& buffer, uint32_t length ) const noexcept;
    void WriteCounter( crc::Buffer& buffer, uint32_t counter ) const noexcept;
    void WriteDataId( crc::Buffer& buffer, uint32_t dataId ) const noexcept;
    void WriteCrc( crc::Buffer& buffer, uint64_t crc ) const noexcept;
    void IncrementCounter() noexcept;

    const Config config;
    uint32_t     counter;
    std::mutex   protectMutex;
};

}  // namespace profile07
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_PROTECTOR07_H_
/* EOF */
