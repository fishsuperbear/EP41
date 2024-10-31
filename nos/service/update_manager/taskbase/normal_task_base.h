/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class CommandTaskBase Header
 */

#ifndef NORMAL_TASK_BASE_H_
#define NORMAL_TASK_BASE_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <vector>
#include <mutex>
#include "diag/libsttask/STNormalTask.h"

using namespace hozon::netaos::sttask;

namespace hozon {
namespace netaos {
namespace update {

    /**
     * @brief Class of NormalTaskBase
     *
     * This class is a normal task.
     */
    class NormalTaskBase : public STNormalTask
    {
    public:
        NormalTaskBase(uint32_t operationId, NormalTaskBase* parent, STObject::TaskCB callback, bool isTopTask);
        virtual ~NormalTaskBase();

        int32_t reqId();

    protected:
        virtual uint32_t doAction();

    int32_t SerializeToBcd(const std::vector<uint8_t>& data, std::string& bcd);

    int32_t ParseFromBcd(const std::string& bcd, std::vector<uint8_t>& data);

    int32_t GetKeyLevel1(uint32_t& key, uint32_t seed, uint32_t APP_MASK);

    int32_t GetKeyLevelFbl(uint32_t& key, uint32_t seed, uint32_t BOOT_MASK);

    uint8_t CalcCrc8(const std::vector<uint8_t>& data, uint8_t crc);

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
    uint16_t CalcCrc16(const std::vector<uint8_t>& data, uint16_t crc);

    private:
        int32_t generateReqId();

    private:
        int32_t m_reqId;

        static uint32_t     s_reqNo;
        static std::mutex   s_sync;

    private:
        NormalTaskBase(const NormalTaskBase&);
        NormalTaskBase& operator=(const NormalTaskBase&);
    };

} // end of update
} // end of netaos
} // end of hozon
#endif  // NORMAL_TASK_BASE_H_
/* EOF */
