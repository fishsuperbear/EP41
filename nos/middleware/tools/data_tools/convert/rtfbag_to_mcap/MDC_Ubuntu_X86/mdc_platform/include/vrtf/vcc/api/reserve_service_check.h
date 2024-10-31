/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 * Description: Add funs to check reserve ServiceId/ServiceName/NodeName for RTFMaintaindService.
 * Create: 2022-05-10
 */
#ifndef VRTF_VCC_INTERNAL_RESERVE_SERVICE_CHECK_H
#define VRTF_VCC_INTERNAL_RESERVE_SERVICE_CHECK_H
#include <set>
#include <string>
#include "vrtf/vcc/api/types.h"
namespace vrtf {
namespace vcc {
namespace rtfmaintaind {
namespace type {
// For RTFMaintaindService to filter EM or PHM service.
namespace ReserveService {
constexpr uint16_t EM_RESERVE_SERVICE_ID_RANGE_BEGIN = 0xe010U;
constexpr uint16_t EM_RESERVE_SERVICE_ID_RANGE_END = 0xe012U;
constexpr uint16_t PHM_RESERVE_SERVICE_ID_RANGE_BEGIN = 0xe020U;
constexpr uint16_t PHM_RESERVE_SERVICE_ID_RANGE_END = 0xe023U;
};
// use by app
constexpr int16_t MAINTAIND_APP_DOMAIN_ID = vrtf::vcc::api::types::reserve::EXTERNAL_SERVICE_DOMAIN_ID;
constexpr uint16_t  MAINTAIND_APP_SERVICE_ID = 0xE000U;
// use by rtftools
constexpr int16_t MAINTAIND_TOOLS_DOMAIN_ID = vrtf::vcc::api::types::reserve::MAINTAIND_SERVICE_DOMAIN_ID;
constexpr uint16_t  MAINTAIND_TOOLS_SERVICE_ID = 0xE001U;

class RtfMaintaindType {
public:
    static std::shared_ptr<RtfMaintaindType> &GetInstance() noexcept
    {
        static std::shared_ptr<RtfMaintaindType> singleInstance{std::make_shared<RtfMaintaindType>()};
        return singleInstance;
    }
    const std::set<std::string>& GetReserveServiceName() const { return reserveServiceName; }
    const std::string& GetMaintaindAppServiceName() const { return maintaindAppServiceName; }
    const std::string& GetMaintaindToolsServiceName() const { return maintaindToolsServiceName; }
private:
    const std::set<std::string> reserveServiceName {
        "ExecutionService",
        "StateService",
        "RecoveryService"
    };
    const std::string maintaindAppServiceName {"/huawei/ap/rtf/RTFMaintaindAppService"};
    const std::string maintaindToolsServiceName {"/huawei/ap/rtf/RTFMaintaindToolsService"};
};
}
}
bool CheckIsReserveServiceId(const uint16_t& checkServiceId) noexcept;
bool CheckIsReserveServiceName(const std::string& checkServiceName) noexcept;
bool CheckIsReserveNodeName(const std::string& checkNodeName) noexcept;
}
}
#endif
