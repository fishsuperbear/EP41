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
#ifndef E2E_INCLUDE_E2EXF_TRANSFORMER_H_
#define E2E_INCLUDE_E2EXF_TRANSFORMER_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include <string>
#include "extend/crc/buffer.h"
#include "ne_someip_e2e_result.h"
#include "extend/e2exf/config.h"
#include "extend/e2exf/types.h"

namespace e2exf {

/// @brief E2E transformer - performs protect and check operations
///        on given data using rules defined in E2E profiles
///
/// @uptrace{SWS_CM_90433}
class Transformer {
   public:
    /// @brief Represents correct result of E2E check (machine state is "Valid", check status is
    /// "Ok")
    static const e2e::Result correctResult;

    /// @brief Constructs transformer with empty configuration
    Transformer();

    /// @brief Constructs transformer with given configuration
    ///
    /// @param cfg    transformer configuration
    explicit Transformer( Config&& cfg );

    /// @brief Constructs transformer with configuration provided as a file on flesystem
    ///
    /// @param filePath               valid path to the confguration file
    /// @param configurationFormat    format of configuration file (JSON or XML)
    explicit Transformer( const std::string& filePath,
                          ConfigurationFormat configurationFormat = ConfigurationFormat::JSON );

    /// @brief Default copy constructor
    Transformer( const Transformer& ) = default;

    /// @brief Default move constructor
    Transformer( Transformer&& ) = default;

    /// @brief Default assignment operator
    Transformer& operator=( const Transformer& ) = default;

    /// @brief Default move-assign operator
    Transformer& operator=( Transformer&& ) = default;

    /// @brief performs E2E check on given buffer, removes E2E header from given buffer and store
    /// the result to output buffer
    ///
    /// @param id           unique data identifier associated with given data
    /// @param inputBuffer  data to be E2E checked
    /// @param outputBuffer inputBuffer with E2E header removed after check
    ///
    /// @throw std::out_of_range is thrown if given buffer size is less than length of SOME/IP
    /// header
    ///
    /// @return Result object containing state and check status
    e2e::Result CheckOutOfPlace( const DataIdentifier id, const crc::Buffer& inputBuffer,
                                 crc::Buffer& outputBuffer );

    /// @brief perform E2E protect on given buffer , adds E2E header to input buffer and store them
    /// to outputBuffer
    ///
    /// @param id           unique data identifier associated with given data
    /// @param inputBuffer  data to be E2E protected
    /// @param outputBuffer inputBuffer with E2E header added after protection
    ///
    /// @throw std::out_of_range is thrown if given buffer size is less than sum of SOME/IP header
    /// length and E2E Profile Header length
    void ProtectOutOfPlace( const DataIdentifier id, const crc::Buffer& inputBuffer,
                            crc::Buffer& outputBuffer );

    /// @brief perform E2E check on given buffer
    ///
    /// @param id      unique data identifier associated with given data
    /// @param buffer  data to be E2E checked
    ///
    /// @return Result object containing state and check status
    e2e::Result Check( const DataIdentifier id, const crc::Buffer& buffer );

    /// @brief perform E2E protect on given buffer, adds E2E header to input buffer
    ///
    /// @param id           unique data identifier associated with given data
    /// @param buffer  data to be E2E protected
    ///
    /// @throw std::invalid_argument is thrown if there is no protector for given data identifier
    /// @throw std::out_of_range is thrown if given buffer size is less than sum of SOME/IP header
    /// length and E2E Profile Header length
    void Protect( const DataIdentifier id, crc::Buffer& buffer );

    /// @brief check if E2E check can be performed for given data identifier
    ///
    /// @param id - unique data identifier associated with given data
    ///
    /// @throw std::invalid-argument is thrown if either profile checker or state machine are not
    /// assotiated with given data identifier
    /// @throw std::out_of_range is thrown if given buffer size is less than sum of SOME/IP header
    /// length and E2E Profile Header length
    ///
    /// @return returns true if for given data ID checker and state machine instances are registered
    bool IsProtected( const DataIdentifier id ) const;

    bool GetCounter(const DataIdentifier id, const crc::Buffer& buffer, std::uint32_t& counter);

#ifdef E2E_DEVELOPMENT
    crc::Buffer lastValidBuffer;
#endif

   private:
    Config config;
    bool retLoadE2EConfig = false;
};

}  // namespace e2exf

#endif  // E2E_INCLUDE_E2EXF_TRANSFORMER_H_
/* EOF */
