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
#ifndef INCLUDE_COM_LIBE2EXF_FAULT_INJECTOR_H
#define INCLUDE_COM_LIBE2EXF_FAULT_INJECTOR_H

#include <string>
#include <vector>
#include <functional>
#include <queue>
#include "extend/crc/buffer.h"

namespace e2exf {

class FaultInjector {
   public:
    using FaultMethod = std::function<bool( crc::Buffer&                    buffer,
                                            std::vector<crc::Buffer>& history )>;

    using FaultMethodWithLog = std::pair<FaultMethod, std::string>;

    static FaultMethod pass;
    static FaultMethod lost;
    static FaultMethod repeat;
    static FaultMethod singleBitCorruption;
    static FaultMethod singleBitCorruptionInE2EHeader;
    static FaultMethod partialDelivery;
    static FaultMethod swapBytes;
    static FaultMethod repeatTwoStepsBefore;
    static FaultMethod Resend( uint32_t historyQueueOffset );

    static std::vector<crc::Buffer> history;

    static void RegisterMethod( FaultMethodWithLog methodWithLog );
    static bool Corrupt( crc::Buffer& inputBuffer );

    static std::queue<FaultMethodWithLog> faultMethodQueue;
};

}  // namespace e2exf

#endif  // INCLUDE_COM_LIBE2EXF_FAULT_INJECTOR_H
