/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: use to generate type-length-value(tlv) buffer and analysis buffer
 * Create: 2020-12-07
 */
#ifndef VRTF_VCC_UTILS_TLV_HELPER_H
#define VRTF_VCC_UTILS_TLV_HELPER_H
#include <mutex>
#include <securec.h>
#include "ara/hwcommon/log/log.h"
/*
Note: Serialize max size is 16M (equal to 0x100000), cannot be result in integer overflow with type of size_t
*/
namespace vrtf {
namespace vcc {
namespace utils {
constexpr std::uint8_t TIME_STAMP_LENGTH = 16U;
constexpr std::uint16_t TIME_STAMP_TYPE_ID = 0xFFFAU;
constexpr size_t TIME_STAMP_SIZE = sizeof(time_t) + sizeof(long);
constexpr size_t TLV_TIME_TOTAL_SIZE = sizeof(TIME_STAMP_LENGTH) + sizeof(TIME_STAMP_TYPE_ID) + TIME_STAMP_SIZE;
enum class TlvType: std::uint8_t {
    TIME_STAMP = 0x00U
};
enum class TlvAnalysisResult: std::uint8_t {
    SUCCESS = 0X00U,
    MEMCPY_FAIL,
    ANALYSIS_FAIL
};
class TlvHelper {
public:
    /**
     * @brief Send event/field By skeleton applications, skeleton should initialize field first.
     * @param[in] buffer data buffer add time stamp info.
     * @param[in] ts user use send interface.
     */
    static void AddTlvTimeStamp(std::uint8_t* buffer, const timespec& ts);

    /**
     * @brief Check data buffer is tlv timeStample label and get send time.
     * @param[in] data receive buffer include tlv time label
     * @param[in] ts analysis tlv time
     * @return TlvAnalysisResult analysis result enum
     */
    static TlvAnalysisResult AnalysisTlvTime(const std::uint8_t * const data, timespec& ts);
};
}
}
}
#endif
