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

#ifndef INCLUDE_COM_LIBE2EXF_DEV_H
#define INCLUDE_COM_LIBE2EXF_DEV_H

#include <cstdint>

#ifndef SHOUT
#define SHOUT \
    std::cout << "\t" << __PRETTY_FUNCTION__ << " :: " << __FILE__ << ":" << __LINE__ << std::endl
#endif

#ifndef SHERR
#define SHERR \
    std::cerr << "\t" << __PRETTY_FUNCTION__ << " :: " << __FILE__ << ":" << __LINE__ << std::endl
#endif

namespace e2exf {

void PrintSomeIpHeader( const uint8_t* data, uint32_t length );
}  // namespace e2exf

#endif  // INCLUDE_COM_LIBE2EXF_DEV_H
