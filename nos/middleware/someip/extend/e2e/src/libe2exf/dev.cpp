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
#include <bitset>
#include <iomanip>
#include <iostream>
#include "extend/crc/buffer.h"
#include "dev.h"

namespace e2exf {

static void Print( const uint8_t* data, const uint32_t length ) {
    for ( uint32_t i{0U}; i < length; ++i ) {
        std::cerr << std::bitset<8>( data[ i ] );
    }
}

static uint32_t PrintUint( const uint8_t* data, const uint32_t length ) {
    uint32_t ret{0U};
    for ( uint32_t i{0U}; i < length; ++i ) {
        ret <<= 8U, ret += static_cast<uint32_t>( data[ i ] );
    }
    return ret;
}

static char const* GetMsgType( uint32_t msgType ) {
    switch ( msgType ) {
        case 0x00:
            return "REQUEST";
        case 0x01U:
            return "REQUEST_NO_RETURN";
        case 0x02U:
            return "NOTIFICATION";
        case 0x40U:
            return "REQUEST ACK";
        case 0x41U:
            return "REQUEST_NO_RETURN ACK";
        case 0x42U:
            return "NOTIFICATION ACK";
        case 0x80U:
            return "RESPONSE";
        case 0x81U:
            return "Error";
        case 0xC0U:
            return "RESPONSE ACK";
        case 0xC1U:
            return "Error ACK";
        default:
            return "== not found ==";
    }
}

#if defined( E2E_DEBUG_PAYLOAD )
static void printPayload( apd::crc::BufferView bufferView ) {
    for ( const auto& b : bufferView ) {
        std::cout << "[" << std::setfill( '0' ) << std::setw( 2 ) << std::hex << (uint32_t) b
                  << std::dec << "]";
    }
    std::cout << std::endl;
}
#endif

void PrintSomeIpHeader( const uint8_t* data, uint32_t /*length*/ ) {
    std::cerr << '\n';
    std::cerr << "MessageID (ServiceId / MethodId):         ";
    Print( data, 4 );
    std::cerr << " " << PrintUint( data, 2 ) << " / " << PrintUint( data + 2U, 2 ) << std::endl;
    std::cerr << "Length:                                   ";
    Print( data + 4, 4 );
    std::cerr << " " << PrintUint( data + 4U, 4 ) << std::endl;
    std::cerr << "RequestId (ClientId / SessionId):         ";
    Print( data + 8, 4 );
    std::cerr << " " << PrintUint( data + 8U, 2 ) << " / " << PrintUint( data + 10U, 2 )
              << std::endl;
    std::cerr << "Protocol / Interface / MsgType / RetCode: ";
    Print( data + 12, 4 );
    std::cerr << " " << PrintUint( data + 12U, 1 ) << " / " << PrintUint( data + 13U, 1 ) << " / "
              << PrintUint( data + 14U, 1 ) << "[" << GetMsgType( PrintUint( data + 14U, 1 ) )
              << "]"
              << " / " << PrintUint( data + 15U, 1 ) << std::endl;
#if defined( E2E_DEBUG_PAYLOAD )
    printPayload( apd::crc::BufferView{data, length} );
#endif
    std::cerr << '\n';
}
}  // namespace e2exf
