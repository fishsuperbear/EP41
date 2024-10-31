/*
* Copyright (c) Hozon Auto Co., Ltd. 2022-2023. All rights reserved.
* Description: security algorithm
*/

#ifndef SECUEITY_ALGORITHM_H
#define SECUEITY_ALGORITHM_H

#include <iostream>
#include <fstream>
#include <string>
#include <mutex>

using namespace std;

namespace hozon {
namespace netaos {
namespace diag {

/********************************************************************************/
class SecurityAlgorithm {
public:

    enum SecurityLevel : uint8_t {
        SecurityLevel_Reserve           = 0x00U,
        SecurityLevel_1_Seed            = 0x03U,
        SecurityLevel_1_Key             = 0x04U,
        SecurityLevel_TEST_Seed         = 0x05U,
        SecurityLevel_TEST_Key          = 0x06U,
        SecurityLevel_FBL_Seed          = 0x11U,
        SecurityLevel_FBL_Key           = 0x12U,
    };

    static SecurityAlgorithm *Instance();

    uint32_t RequestSecurityAlgorithm(const uint8_t level, const uint32_t mask, const uint32_t seed = 0);

private:
    uint32_t GetRandomNum();
    uint32_t SecurityAlgorithmLevel1(const uint32_t seed, const uint32_t mask);
    uint32_t SecurityAlgorithmLevelTest(const uint32_t seed, const uint32_t mask);
    uint32_t SecurityAlgorithmLevelFBL(const uint32_t seed, const uint32_t mask);

private:
    SecurityAlgorithm();
    ~SecurityAlgorithm();
    SecurityAlgorithm(const SecurityAlgorithm &);
    SecurityAlgorithm & operator = (const SecurityAlgorithm &);

    static SecurityAlgorithm *instancePtr_;
    static mutex instance_mtx_;
    static mutex mtx_;

};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // SECUEITY_ALGORITHM_H
