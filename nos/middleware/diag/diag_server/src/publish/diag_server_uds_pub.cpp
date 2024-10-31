#include "ara/core/string_view.h"
#include <ara/core/initialization.h>
#include "socudsservice_skeleton_define.h"
#include "diag/diag_server/include/common/diag_server_logger.h"
#include "diag/diag_server/include/publish/diag_server_uds_pub.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerUdsPub* DiagServerUdsPub::instance_ = nullptr;
std::mutex DiagServerUdsPub::mtx_;

DiagServerUdsPub*
DiagServerUdsPub::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerUdsPub();
        }
    }
    return instance_;
}

DiagServerUdsImpl::DiagServerUdsImpl(ara::com::InstanceIdentifier instanceID, ara::com::MethodCallProcessingMode mode, DiagServerUdsPub* handle)
: Skeleton(instanceID, mode)
{
    DG_INFO << "DiagServerUdsImpl::DiagServerUdsImpl";
    if (nullptr != handle) {
        handle_ = handle;
    }
}

DiagServerUdsImpl::DiagServerUdsImpl(ara::core::InstanceSpecifier instance_specifier, ara::com::MethodCallProcessingMode mode, DiagServerUdsPub* handle)
: Skeleton(instance_specifier, mode)
{
    DG_INFO << "DiagServerUdsImpl::DiagServerUdsImpl";
    if (nullptr != handle) {
        handle_ = handle;
    }
}

DiagServerUdsImpl::~DiagServerUdsImpl()
{
    DG_INFO << "DiagServerUdsImpl::~DiagServerUdsImpl";
}

ara::core::Future<methods::SoCUdsService::McuUdsRes::Output>
DiagServerUdsImpl::McuUdsRes(const ::hozon::netaos::McuDiagDataType& McuDiagData)
{
    DG_INFO << "DiagServerUdsImpl::McuUdsRes";
    ara::core::Promise<methods::SoCUdsService::McuUdsRes::Output> promise;
    methods::SoCUdsService::McuUdsRes::Output value;
    std::vector<uint8_t> udsres;
    uint32_t udssize = ((uint32_t)(McuDiagData.DIDheader[0]) << 24) | ((uint32_t)(McuDiagData.DIDheader[1]) << 16) | ((uint32_t)(McuDiagData.DIDheader[2]) << 8) | (uint32_t)McuDiagData.DIDheader[3];
    if (udssize != sizeof(McuDiagData) - 4) {
        DG_ERROR << "DiagServerUdsImpl::McuUdsRes length invalid udssize: " << udssize << ", sizeof(McuDiagData): " << sizeof(McuDiagData) << ", uds service: " << McuDiagData.DIDheader[4];
        value.McuResult = hozon::netaos::mcuResultEnum::Failed;
        promise.set_value(value);
        return promise.get_future();
    }

    udsres.resize(udssize);
    memcpy(udsres.data(), &McuDiagData.DIDheader[4], udssize);
    if (handle_->OnMcuUdsRes(udsres)) {
        DG_ERROR << "DiagServerUdsImpl::McuUdsRes OnMcuUdsRes failed " << udssize << ", sizeof(McuDiagData): " << sizeof(McuDiagData) << ", uds service: " << McuDiagData.DIDheader[4];
        value.McuResult = hozon::netaos::mcuResultEnum::Failed;
        promise.set_value(value);
        return promise.get_future();
    }

    value.McuResult = hozon::netaos::mcuResultEnum::Success;
    promise.set_value(value);
    return promise.get_future();
}

DiagServerUdsPub::DiagServerUdsPub()
{
    map_mcu_did_info_[0xF1C1] = std::make_shared<std::vector<uint8_t>>(16, 0);
    map_mcu_did_info_[0xF1C2] = std::make_shared<std::vector<uint8_t>>(64, 0);
    map_mcu_did_info_[0xF1C3] = std::make_shared<std::vector<uint8_t>>(64, 0);
    map_mcu_did_info_[0xF1C4] = std::make_shared<std::vector<uint8_t>>(64, 0);
    map_mcu_did_info_[0xF1C5] = std::make_shared<std::vector<uint8_t>>(64, 0);
    map_mcu_did_info_[0xF1C6] = std::make_shared<std::vector<uint8_t>>(64, 0);
}

void
DiagServerUdsPub::Init()
{
    DG_INFO << "DiagServerUdsPub::Init";
    std::thread diag_to_mcu([this]() {
        ara::core::Initialize();
        skeleton_ = std::make_shared<DiagServerUdsImpl>(ara::com::InstanceIdentifier("1"), ara::com::MethodCallProcessingMode::kEvent, this);
        if (nullptr != skeleton_) {
            DG_INFO << "DiagServerUdsPub::Init OfferService";
            skeleton_->OfferService();
        }
    });

    pthread_setname_np(diag_to_mcu.native_handle(), "diag_to_mcu");
    diag_to_mcu.detach();
    DG_INFO << "DiagServerUdsPub::Init finish!";
}

void
DiagServerUdsPub::DeInit()
{
    DG_INFO << "DiagServerUdsPub::DeInit";
    // 停止服务
    if (nullptr != skeleton_) {
        skeleton_->StopOfferService();
    }
    DG_INFO << "DiagServerUdsPub::DeInit finish!";
}

bool
DiagServerUdsPub::GetMcuDidsInfo(uint16_t did, std::vector<uint8_t>& uds)
{
    DG_INFO << "DiagServerUdsPub::GetMcuDidsInfo did: 0x" << std::hex << did;
    std::vector<uint8_t> uds_req;
    uds_req.push_back(0x03);
    uds_req.push_back(0x22);
    uds_req.push_back((uint8_t)(did >> 8));
    uds_req.push_back((uint8_t)(did));
    SendUdsEvent(uds_req);

    for (auto &it : map_mcu_did_info_) {
        if (did == it.first) {
            uds.clear();
            uds.insert(uds.end(), it.second->data(), it.second->data() + it.second->size());
        }
    }

    return (uds.size() > 0) ? true : false;
}

bool
DiagServerUdsPub::SendUdsEvent(std::vector<uint8_t> uds)
{
    DG_INFO << "DiagServerUdsPub::SendUdsEvent start ";
    if (skeleton_ == nullptr) {
        DG_ERROR << "DiagServerUdsPub::SendUdsEvent skeleton_ is NULL";
        return false;
    }

    if (uds.size() < 2) {
        DG_ERROR << "DiagServerUdsPub::SendUdsEvent uds data invalid";
        return false;
    }

    hozon::netaos::SocUdsReqData data;
    data = { 0x00 };
    uint32_t i = 0;
    for (auto itr : uds) {
        data[i] = itr;
        ++i;
    }

    skeleton_->SocUdsReq.Send(data);

    return (uds.size() > 2) ? true : false;
}

bool
DiagServerUdsPub::OnMcuUdsRes(const std::vector<uint8_t>& uds)
{
    DG_INFO << "DiagServerUdsPub::OnMcuUdsRes  size: " << uds.size();
    for (auto it = uds.begin(); uds.end() != it; ++it) {
        uint8_t service = uds[0];
        switch (service) {
        case 0x62:  // response ECU DIDS read
            {
                if (uds.size() > 3) {
                    return UdsDidsParse(uds);
                }
            }
            break;
        default:
            break;
        }
    }
    return false;
}

bool
DiagServerUdsPub::UdsDidsParse(const std::vector<uint8_t>& uds)
{
    for (uint32_t index = 1; index < uds.size(); ++index) {
        if (index + 2 > uds.size()) {
            return false;
        }
        uint16_t did = (uds[index] << 8 | uds[index + 1]);
        switch (did) {
        case 0xF1C1:
            {
                if (index + 2  + 16 >  uds.size() ) {
                    memcpy(map_mcu_did_info_[did]->data(), &uds[index + 2], uds.size() - index - 2);
                }
                else {
                    memcpy(map_mcu_did_info_[did]->data(), &uds[index + 2], 16);
                }
                index += 17;
            }
            break;
        case 0xF1C2:
        case 0xF1C3:
        case 0xF1C4:
        case 0xF1C5:
        case 0xF1C6:
            {
                if (index + 2  + 64 >  uds.size() ) {
                    memcpy(map_mcu_did_info_[did]->data(), &uds[index + 2], uds.size() - index - 2);
                } else {
                    memcpy(map_mcu_did_info_[did]->data(), &uds[index + 2], 64);
                }
                index += 65;
            }
            break;
        default:
            break;
        }
    }

    return (uds.size() > 0) ? true : false;
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
