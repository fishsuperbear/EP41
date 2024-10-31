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
#include "extend/e2e/profile_04.h"
#include "extend/crc/crc.h"

namespace profile {
namespace profile04 {

Config::Config( uint32_t curDataId, uint16_t curHeaderOffset, uint16_t curMinDataLength,
                uint16_t curMaxDataLength, uint16_t curMaxDeltaCounter )
    : dataId{curDataId}
    , offset{curHeaderOffset}
    , minDataLength{curMinDataLength}
    , maxDataLength{curMaxDataLength}
    , maxDeltaCounter{curMaxDeltaCounter} {
    if ( maxDataLength > Profile04::maxCrcProtectedDataLength ) {
        throw std::invalid_argument{
            "Wrong configuration of E2E: maxDataLength > "
            "Profile04Specification::maxCrcProtectedDataLength"};
    }
    if ( minDataLength > maxDataLength ) {
        throw std::invalid_argument{"Wrong configuration of E2E: minDataLength > maxDataLength"};
    }
    if ( offset + Profile04::headerLength > maxDataLength ) {
        throw std::invalid_argument{
            "Wrong configuration of E2E: offset + Profile04Specification::headerLength > "
            "maxDataLength"};
    }
    if ( offset + Profile04::headerLength < minDataLength ) {
        throw std::invalid_argument{
            "Wrong configuration of E2E: offset + Profile04Specification::headerLength < "
            "minDataLength"};
    }
}

uint32_t Profile04::ComputeCrc( const Config&                config,
                                const crc::Buffer& buffer ) noexcept {
    using crc::CRC;
    // calculate crc on first part of buffer
    uint32_t computedCRC =
        CRC::CalculateCRC32P4( crc::BufferView{buffer, 0U, config.offset + 8U} );
    // continue after the 4-byte gap
    return CRC::CalculateCRC32P4(
        crc::BufferView{buffer, config.offset + 12U, buffer.size()}, false, computedCRC );
}

/** \uptrace{SWS_E2E_00356} */
bool Profile04::IsBufferLengthValid( const Config&                config,
                                     const crc::Buffer& buffer ) noexcept {
    return ( ( config.minDataLength <= buffer.size() ) &&
             ( buffer.size() <= config.maxDataLength ) );
}
}  // namespace profile04
}  // namespace profile
/* EOF */
