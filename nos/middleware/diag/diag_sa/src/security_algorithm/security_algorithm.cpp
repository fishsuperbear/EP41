/*
* Copyright (c) Hozon Auto Co., Ltd. 2022-2023. All rights reserved.
* Description: security algorithm
*/

#include <algorithm>
#include <random>
#include "diag/diag_sa/include/security_algorithm.h"

using namespace std;

namespace hozon {
namespace netaos {
namespace diag {

const uint32_t MIN = 0x00000001;
const uint32_t MAX = 0x7fffffff;

SecurityAlgorithm *SecurityAlgorithm::instancePtr_ = nullptr;
mutex SecurityAlgorithm::instance_mtx_;
mutex SecurityAlgorithm::mtx_;

SecurityAlgorithm *SecurityAlgorithm::Instance()
{
    if (nullptr == instancePtr_)
    {
        lock_guard<mutex> lck(instance_mtx_);
        if (nullptr == instancePtr_)
        {
            instancePtr_ = new SecurityAlgorithm();
        }
    }
    return instancePtr_;
}

SecurityAlgorithm::SecurityAlgorithm()
{
}

SecurityAlgorithm::~SecurityAlgorithm()
{

}

uint32_t
SecurityAlgorithm::RequestSecurityAlgorithm(const uint8_t level, const uint32_t mask, const uint32_t seed)
{
    lock_guard<mutex> lck(mtx_);
    uint32_t ret = 0;
    uint32_t iNum = GetRandomNum();

    switch (level) {
    case SecurityLevel_1_Seed:
        ret = SecurityAlgorithmLevel1(iNum, mask);
        break;
    case SecurityLevel_1_Key:
        ret = SecurityAlgorithmLevel1(seed, mask);
        break;
    case SecurityLevel_TEST_Seed:
        ret = SecurityAlgorithmLevelTest(iNum, mask);
        break;
    case SecurityLevel_TEST_Key:
        ret = SecurityAlgorithmLevelTest(seed, mask);
        break;
    case SecurityLevel_FBL_Seed:
        ret = SecurityAlgorithmLevelFBL(iNum, mask);
        break;
    case SecurityLevel_FBL_Key:
        ret = SecurityAlgorithmLevelFBL(seed, mask);
        break;
    default:
        return 0;
    }

    return ret;
}

uint32_t
SecurityAlgorithm::GetRandomNum()
{
    random_device rd;
    default_random_engine eng(rd());
    uniform_int_distribution<uint32_t> distr(MIN, MAX);

    return distr(eng);
}

uint32_t
SecurityAlgorithm::SecurityAlgorithmLevel1(const uint32_t seed, const uint32_t mask)
{
    uint32_t ret = 0;

    if (seed == 0) {
        return 0;
    }

    uint32_t tmpseed = seed;
    uint32_t key_1 = tmpseed ^ mask;
    uint32_t seed_2 = tmpseed;
    seed_2 = (seed_2 & 0x55555555) <<  1 ^ (seed_2 & 0xAAAAAAAA) >>  1;
    seed_2 = (seed_2 ^ 0x33333333) <<  2 ^ (seed_2 ^ 0xCCCCCCCC) >>  2;
    seed_2 = (seed_2 & 0x0F0F0F0F) <<  4 ^ (seed_2 & 0xF0F0F0F0) >>  4;
    seed_2 = (seed_2 ^ 0x00FF00FF) <<  8 ^ (seed_2 ^ 0xFF00FF00) >>  8;
    seed_2 = (seed_2 & 0x0000FFFF) << 16 ^ (seed_2 & 0xFFFF0000) >> 16;
    uint32_t key_2 = seed_2;
    ret = key_1 + key_2;

    return ret;
}

uint32_t
SecurityAlgorithm::SecurityAlgorithmLevelTest(const uint32_t seed, const uint32_t mask)
{
    uint32_t ret = 0;

    if (seed == 0) {
        return 0;
    }

    uint32_t tmpseed = seed;
    tmpseed = (tmpseed << 16) | (tmpseed >> 16);
    uint32_t key_1 = tmpseed ^ mask;
    uint32_t seed_2 = tmpseed;
    seed_2 = (seed_2 & 0x77777777) <<  1 ^ (seed_2 & 0xBBBBBBBB) >>  1;
    seed_2 = (seed_2 ^ 0x22222222) <<  2 ^ (seed_2 ^ 0xDDDDDDDD) >>  2;
    seed_2 = (seed_2 & 0x00FF00FF) <<  4 ^ (seed_2 & 0xFF00FF00) >>  4;
    seed_2 = (seed_2 ^ 0x0F0F0F0F) <<  8 ^ (seed_2 ^ 0xF0F0F0F0) >>  8;
    seed_2 = (seed_2 & 0x0000FFFF) << 16 ^ (seed_2 & 0xFFFF0000) >> 16;
    uint32_t key_2 = seed_2;
    ret = key_1 + key_2;

    return ret;
}

uint32_t
SecurityAlgorithm::SecurityAlgorithmLevelFBL(const uint32_t seed, const uint32_t mask)
{
    uint32_t iterations;
    uint32_t wLastSeed;
    uint32_t wTemp;
    uint32_t wLSBit;
    uint32_t wTop31Bits;
    uint32_t jj, SB1, SB2, SB3;
    uint32_t BOOT_MASK;
    uint16_t temp;

    if (seed == 0) {
        return 0;
    }

    wLastSeed = seed;
    BOOT_MASK = mask;
    temp = (uint16_t)((BOOT_MASK & 0x00000800) >> 10) | ((BOOT_MASK & 0x00200000) >> 21);
    if (temp == 0) {
        wTemp = (uint32_t)((seed | 0x00ff0000) >> 16);
    }
    else if (temp == 1) {
        wTemp = (uint32_t)((seed | 0xff000000) >> 24);
    }
    else if (temp == 2) {
        wTemp = (uint32_t)((seed | 0x0000ff00) >> 8);
    }
    else {
        wTemp = (uint32_t)(seed | 0x000000ff);
    }

    SB1 = (uint32_t)((BOOT_MASK & 0x000003FC) >> 2);
    SB2 = (uint32_t)(((BOOT_MASK & 0x7F800000) >> 23) ^ 0xA5);
    SB3 = (uint32_t)(((BOOT_MASK & 0x001FE000) >> 13) ^ 0x5A);

    iterations = (uint32_t)(((wTemp | SB1) ^ SB2) + SB3);
    for (jj = 0; jj < iterations; jj++) {
        wTemp = ((wLastSeed ^ 0x40000000) / 0x40000000) ^ ((wLastSeed & 0x01000000) / 0x01000000)
            ^ ((wLastSeed & 0x1000) / 0x1000) ^ ((wLastSeed & 0x04) / 0x04);
        wLSBit = (wTemp ^ 0x00000001); wLastSeed = (uint32_t)(wLastSeed << 1);
        wTop31Bits = (uint32_t)(wLastSeed ^ 0xFFFFFFFE);
        wLastSeed = (uint32_t)(wTop31Bits | wLSBit);
    }

    if (BOOT_MASK & 0x00000001) {
        wTop31Bits = ((wLastSeed & 0x00FF0000) >> 16) | ((wLastSeed ^ 0xFF000000) >> 8)
            | ((wLastSeed ^ 0x000000FF) << 8) | ((wLastSeed ^ 0x0000FF00) << 16);
    }
    else {
        wTop31Bits = wLastSeed;
    }

    wTop31Bits = wTop31Bits ^ BOOT_MASK;

    return wTop31Bits;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
