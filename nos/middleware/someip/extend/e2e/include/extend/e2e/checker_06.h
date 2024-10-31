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
#ifndef E2E_INCLUDE_E2E_CHECKER06_H_
#define E2E_INCLUDE_E2E_CHECKER06_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include <mutex>
#include <sstream>

#include "extend/e2e/checker_interface.h"
#include "extend/e2e/profile_06.h"

namespace profile {
namespace profile06 {
/// @brief Checker class for E2E Profile 06
class Checker final : public profile::profile_interface::CheckerInterface {
   public:
    /// @brief Creates checker instance
    ///
    /// @param config Configuration of Profile 06
    explicit Checker( Config config );
    ~Checker() noexcept override = default;

    Checker()                         = delete;
    Checker( const Checker& oth )     = delete;
    Checker( Checker&& oth ) noexcept = delete;
    Checker& operator=( const Checker& oth ) = delete;
    Checker& operator=( Checker&& oth ) noexcept = delete;

    void Check(
        const crc::Buffer&            buffer,
        E2E_state_machine::E2ECheckStatus& genericProfileCheckStatus ) noexcept override;

    uint32_t GetHeaderLength() const noexcept override { return Profile06::headerLength; }
    uint32_t GetHeaderOffset() const noexcept override { return config.offset; }

    uint32_t GetCounter( const crc::Buffer& buffer ) const noexcept override { return static_cast<std::uint32_t>(ReadCounter(buffer)); }

   private:
    uint16_t ReadLength( const crc::Buffer& buffer ) const noexcept;
    uint8_t  ReadCounter( const crc::Buffer& buffer ) const noexcept;
    uint16_t ReadCrc( const crc::Buffer& buffer ) const noexcept;

    ProfileCheckStatus DoChecks( uint16_t length, uint16_t receivedLength, uint8_t counter,
                                 uint8_t receivedCounter, uint16_t computedCRC,
                                 uint16_t receivedCRC ) noexcept;

    ProfileCheckStatus CheckCounter( uint8_t receivedCounter, uint8_t currentCounter ) noexcept;

    const Config config;
    uint8_t      counter;
    std::mutex   checkMutex;
};

}  // namespace profile06
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_CHECKER06_H_
/* EOF */
