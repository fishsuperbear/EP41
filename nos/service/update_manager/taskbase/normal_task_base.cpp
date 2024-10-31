/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class NormakTaskBase implement
 */


#include "update_manager/taskbase/normal_task_base.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

    uint32_t NormalTaskBase::s_reqNo = 0;
    std::mutex NormalTaskBase::s_sync;

    NormalTaskBase::NormalTaskBase(uint32_t operationId,
                                    NormalTaskBase* parent,
                                    STObject::TaskCB callback,
                                    bool isTopTask)
            : STNormalTask(operationId, parent, callback, isTopTask)
            , m_reqId(0)
    {
        if (isTopTask) {
            m_reqId = generateReqId();
        }
        else {
            if (parent) {
                m_reqId = parent->reqId();
            }
        }
    }

    NormalTaskBase::~NormalTaskBase()
    {
    }

    int32_t NormalTaskBase::generateReqId()
    {
        std::lock_guard<std::mutex> sync(s_sync);
        return (++s_reqNo > 0) ? s_reqNo : s_reqNo = 1;
    }

    int32_t NormalTaskBase::reqId()
    {
        return m_reqId;
    }

    uint32_t NormalTaskBase::doAction()
    {
        return eContinue;
    }

    int32_t NormalTaskBase::SerializeToBcd(const std::vector<uint8_t>& data, std::string& bcd)
    {
        int32_t ret = -1;
        if (0 == data.size()) {
            return ret;
        }

        uint8_t char_index[] = { "0123456789ABCDEF" };
        for (auto &it : data) {
            bcd.push_back(char_index[it >> 4]);
            bcd.push_back(char_index[it & 0x0F]);
        }

        return bcd.size();
    }

    int32_t NormalTaskBase::ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data)
    {
        int32_t ret = -1;
        if (0 == bcd.size()) {
            return ret;
        }
        std::string tmp = bcd;
        if (bcd.size() %2) {
            tmp = std::string("0") + bcd;
        }

        for (uint16_t index = 0; index < bcd.size(); index = index  + 2) {
            uint8_t left = tmp[index];
            uint8_t right = tmp[index + 1];
            if (left >= '0' && left <= '9') {
                left -= '0';
            }
            else if (left >= 'A' && left <= 'F') {
                left -= 'A' - 10;
            }
            else if (left >= 'a' && left <= 'f') {
                left -= 'a' - 10;
            }
            else {
                break;
            }

            if (right >= '0' && right <= '9') {
                right -= '0';
            }
            else if (right >= 'A' && right <= 'F') {
                right -= 'A' - 10;
            }
            else if (right >= 'a' && right <= 'f') {
                right -= 'a' - 10;
            }
            else {
                break;
            }
            data.push_back(left << 4 | right);
        }

        return data.size();
    }

    int32_t NormalTaskBase::GetKeyLevel1(uint32_t& key, uint32_t seed, uint32_t APP_MASK)
    {
        int32_t ret = -1;
        if (seed == 0) {
            return 0;
        }
        uint32_t tmpseed = seed;
        uint32_t key_1 = tmpseed ^ APP_MASK;
        uint32_t seed_2 = tmpseed;
        seed_2 = (seed_2 & 0x55555555) << 1 ^ (seed_2 & 0xAAAAAAAA) >> 1;
        seed_2 = (seed_2 ^ 0x33333333) << 2 ^ (seed_2 ^ 0xCCCCCCCC) >> 2;
        seed_2 = (seed_2 & 0x0F0F0F0F) << 4 ^ (seed_2 & 0xF0F0F0F0) >> 4;
        seed_2 = (seed_2 ^ 0x00FF00FF) << 8 ^ (seed_2 ^ 0xFF00FF00) >> 8;
        seed_2 = (seed_2 & 0x0000FFFF) << 16 ^ (seed_2 & 0xFFFF0000) >> 16;
        uint32_t key_2 = seed_2;
        key = key_1 + key_2;
        ret = key;
        return ret;
    }

    int32_t NormalTaskBase::GetKeyLevelFbl(uint32_t& key, uint32_t seed, uint32_t BOOT_MASK)
    {
        int32_t ret = -1;
        if (seed == 0) {
            return 0;
        }

        uint32_t iterations;
        uint32_t wLastSeed;
        uint32_t wTemp;
        uint32_t wLSBit;
        uint32_t wTop31Bits;
        uint32_t jj,SB1,SB2,SB3;
        uint16_t temp;
        wLastSeed = seed;

        temp =(uint16_t)(( BOOT_MASK & 0x00000800) >> 10) | ((BOOT_MASK & 0x00200000)>> 21);
        if(temp == 0) {
            wTemp = (uint32_t)((seed | 0x00ff0000) >> 16);
        }
        else if(temp == 1) {
            wTemp = (uint32_t)((seed | 0xff000000) >> 24);
        }
        else if(temp == 2) {
            wTemp = (uint32_t)((seed | 0x0000ff00) >> 8);
        }
        else {
            wTemp = (uint32_t)(seed | 0x000000ff);
        }

        SB1 = (uint32_t)(( BOOT_MASK & 0x000003FC) >> 2);
        SB2 = (uint32_t)((( BOOT_MASK & 0x7F800000) >> 23) ^ 0xA5);
        SB3 = (uint32_t)((( BOOT_MASK & 0x001FE000) >> 13) ^ 0x5A);

        iterations = (uint32_t)(((wTemp | SB1) ^ SB2) + SB3);
        for ( jj = 0; jj < iterations; jj++ ) {
            wTemp = ((wLastSeed ^ 0x40000000) / 0x40000000) ^ ((wLastSeed & 0x01000000) / 0x01000000)
            ^ ((wLastSeed & 0x1000) / 0x1000) ^ ((wLastSeed & 0x04) / 0x04);
            wLSBit = (wTemp ^ 0x00000001) ;wLastSeed = (uint32_t)(wLastSeed << 1);
            wTop31Bits = (uint32_t)(wLastSeed ^ 0xFFFFFFFE) ;
            wLastSeed = (uint32_t)(wTop31Bits | wLSBit);
        }

        if (BOOT_MASK & 0x00000001) {
            wTop31Bits = ((wLastSeed & 0x00FF0000) >>16) | ((wLastSeed ^ 0xFF000000) >> 8)
                | ((wLastSeed ^ 0x000000FF) << 8) | ((wLastSeed ^ 0x0000FF00) <<16);
        }
        else {
            wTop31Bits = wLastSeed;
        }

        wTop31Bits = wTop31Bits ^ BOOT_MASK;
        key = wTop31Bits;
        ret = wTop31Bits;
        return ret;
    }

    uint8_t NormalTaskBase::CalcCrc8(const std::vector<uint8_t>& data, uint8_t crc)
    {
        uint8_t crc8 = crc;
        for (auto it: data) {
            crc8 += it;
        }
        return crc8;
    }

    /*********************************************************
     *
     * The checksum algorithm to be used shall be the CRC16-CITT:
     *  - Polynomial: x^16+x^12+x^5+1 (1021 hex)
     *  - Initial value: FFFF (hex)
     *  For a fast CRC16-CITT calculation a look-up table implementation is the preferred solution. For ECUs with a
     *  limited amount of flash memory (or RAM), other implementations may be necessary.
     *  Example 1: crc16-citt c-code (fast)
     *  This example uses a look-up table with pre-calculated CRCs for fast calculation.
     * ******************************************************/
    uint16_t NormalTaskBase::CalcCrc16(const std::vector<uint8_t>& data, uint16_t crc)
    {
        /*Here is crctab[256], this array is fixed */
        uint16_t crctab[256] =
        {
            0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50A5, 0x60C6, 0x70E7,
            0x8108, 0x9129, 0xA14A, 0xB16B, 0xC18C, 0xD1AD, 0xE1CE, 0xF1EF,
            0x1231, 0x0210, 0x3273, 0x2252, 0x52B5, 0x4294, 0x72F7, 0x62D6,
            0x9339, 0x8318, 0xB37B, 0xA35A, 0xD3BD, 0xC39C, 0xF3FF, 0xE3DE,
            0x2462, 0x3443, 0x0420, 0x1401, 0x64E6, 0x74C7, 0x44A4, 0x5485,
            0xA56A, 0xB54B, 0x8528, 0x9509, 0xE5EE, 0xF5CF, 0xC5AC, 0xD58D,
            0x3653, 0x2672, 0x1611, 0x0630, 0x76D7, 0x66F6, 0x5695, 0x46B4,
            0xB75B, 0xA77A, 0x9719, 0x8738, 0xF7DF, 0xE7FE, 0xD79D, 0xC7BC,
            0x48C4, 0x58E5, 0x6886, 0x78A7, 0x0840, 0x1861, 0x2802, 0x3823,
            0xC9CC, 0xD9ED, 0xE98E, 0xF9AF, 0x8948, 0x9969, 0xA90A, 0xB92B,
            0x5AF5, 0x4AD4, 0x7AB7, 0x6A96, 0x1A71, 0x0A50, 0x3A33, 0x2A12,
            0xDBFD, 0xCBDC, 0xFBBF, 0xEB9E, 0x9B79, 0x8B58, 0xBB3B, 0xAB1A,
            0x6CA6, 0x7C87, 0x4CE4, 0x5CC5, 0x2C22, 0x3C03, 0x0C60, 0x1C41,
            0xEDAE, 0xFD8F, 0xCDEC, 0xDDCD, 0xAD2A, 0xBD0B, 0x8D68, 0x9D49,
            0x7E97, 0x6EB6, 0x5ED5, 0x4EF4, 0x3E13, 0x2E32, 0x1E51, 0x0E70,
            0xFF9F, 0xEFBE, 0xDFDD, 0xCFFC, 0xBF1B, 0xAF3A, 0x9F59, 0x8F78,
            0x9188, 0x81A9, 0xB1CA, 0xA1EB, 0xD10C, 0xC12D, 0xF14E, 0xE16F,
            0x1080, 0x00A1, 0x30C2, 0x20E3, 0x5004, 0x4025, 0x7046, 0x6067,
            0x83B9, 0x9398, 0xA3FB, 0xB3DA, 0xC33D, 0xD31C, 0xE37F, 0xF35E,
            0x02B1, 0x1290, 0x22F3, 0x32D2, 0x4235, 0x5214, 0x6277, 0x7256,
            0xB5EA, 0xA5CB, 0x95A8, 0x8589, 0xF56E, 0xE54F, 0xD52C, 0xC50D,
            0x34E2, 0x24C3, 0x14A0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
            0xA7DB, 0xB7FA, 0x8799, 0x97B8, 0xE75F, 0xF77E, 0xC71D, 0xD73C,
            0x26D3, 0x36F2, 0x0691, 0x16B0, 0x6657, 0x7676, 0x4615, 0x5634,
            0xD94C, 0xC96D, 0xF90E, 0xE92F, 0x99C8, 0x89E9, 0xB98A, 0xA9AB,
            0x5844, 0x4865, 0x7806, 0x6827, 0x18C0, 0x08E1, 0x3882, 0x28A3,
            0xCB7D, 0xDB5C, 0xEB3F, 0xFB1E, 0x8BF9, 0x9BD8, 0xABBB, 0xBB9A,
            0x4A75, 0x5A54, 0x6A37, 0x7A16, 0x0AF1, 0x1AD0, 0x2AB3, 0x3A92,
            0xFD2E, 0xED0F, 0xDD6C, 0xCD4D, 0xBDAA, 0xAD8B, 0x9DE8, 0x8DC9,
            0x7C26, 0x6C07, 0x5C64, 0x4C45, 0x3CA2, 0x2C83, 0x1CE0, 0x0CC1,
            0xEF1F, 0xFF3E, 0xCF5D, 0xDF7C, 0xAF9B, 0xBFBA, 0x8FD9, 0x9FF8,
            0x6E17, 0x7E36, 0x4E55, 0x5E74, 0x2E93, 0x3EB2, 0x0ED1, 0x1EF0
        };

        uint16_t crc16 = crc;
        uint16_t tmp = 0;
        for (auto it : data) {
            tmp = (crc16 >> 8) ^ it;
            crc16 = (crc16 << 8) ^ crctab[tmp];
        }

        return crc16;
    }

} // end of update
} // end of netaos
} // end of hozon
/* EOF */
