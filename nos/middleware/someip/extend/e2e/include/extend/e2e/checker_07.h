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
#ifndef E2E_INCLUDE_E2E_CHECKER07_H_
#define E2E_INCLUDE_E2E_CHECKER07_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <cstdint>
#include <mutex>

#include "extend/e2e/checker_interface.h"
#include "extend/e2e/profile_07.h"

namespace profile {
namespace profile07 {

/// @brief Checker class for E2E Profile 07
class Checker final : public profile::profile_interface::CheckerInterface {
   public:
    /// @brief Creates checker instance
    ///
    /// @param config Configuration of Profile 07
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

    uint32_t GetHeaderLength() const noexcept override { return Profile07::headerLength; }
    uint32_t GetHeaderOffset() const noexcept override { return config.offset; }

    uint32_t GetCounter( const crc::Buffer& buffer ) const noexcept override { return ReadCounter(buffer); }

   private:
    uint32_t ReadLength( const crc::Buffer& buffer ) const noexcept;
    uint32_t ReadCounter( const crc::Buffer& buffer ) const noexcept;
    uint32_t ReadDataId( const crc::Buffer& buffer ) const noexcept;
    uint64_t ReadCrc( const crc::Buffer& buffer ) const noexcept;

    ProfileCheckStatus DoChecks( uint32_t length, uint32_t receivedLength, uint32_t counter,
                                 uint32_t receivedCounter, uint32_t dataId, uint32_t receivedDataId,
                                 uint64_t computedCRC, uint64_t receivedCRC ) noexcept;

    ProfileCheckStatus CheckCounter( uint32_t receivedCounter, uint32_t currentCounter ) noexcept;

    const Config config;
    uint32_t     counter;
    std::mutex   checkMutex;
};

}  // namespace profile07
}  // namespace profile

#endif  // E2E_INCLUDE_E2E_CHECKER07_H_
/* EOF */
