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
#ifndef E2E_INCLUDE_E2E_CHECKER22_H_
#define E2E_INCLUDE_E2E_CHECKER22_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "extend/e2e/checker_interface.h"
#include "extend/e2e/profile_22.h"

namespace profile {
namespace profile22 {

/// @brief Checker class for E2E Profile 22
class Checker final : public profile::profile_interface::CheckerInterface {
   public:
    Checker() = delete;

    /// @brief Creates checker instance
    ///
    /// @param config Configuration of Profile 22
    explicit Checker( Config config );
    Checker( const Checker& )     = delete;
    Checker( Checker&& ) noexcept = delete;
    Checker& operator=( const Checker& ) = delete;
    Checker& operator=( Checker&& ) noexcept = delete;
    ~Checker() noexcept override             = default;

    void Check(
        const crc::Buffer&            buffer,
        E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) noexcept override;

    uint32_t GetHeaderLength() const noexcept override { return Profile22::headerLength; }
    uint32_t GetHeaderOffset() const noexcept override { return 0U; }

    uint32_t GetCounter( const crc::Buffer& buffer ) const noexcept override { return static_cast<std::uint32_t>(ReadCounter(buffer)); }

   private:
    uint8_t ReadCounter( const crc::Buffer& buffer ) const noexcept;
    uint8_t ReadCrc( const crc::Buffer& buffer ) const noexcept;

    ProfileCheckStatus DoChecks( uint8_t currentCounter, uint8_t receivedCounter,
                                 uint8_t computedCrc, uint8_t receivedCrc ) const;

    uint8_t CalculateCounterDelta( uint8_t receivedCounter, uint8_t currentCounter ) const noexcept;

    const Config config;
    uint8_t      counter : 4;
    std::mutex   checkMutex;
};

}  // namespace profile22
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_CHECKER22_H_
/* EOF */
