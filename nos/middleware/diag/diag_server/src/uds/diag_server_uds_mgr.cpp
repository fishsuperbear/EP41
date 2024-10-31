#include "diag/diag_server/include/uds/diag_server_uds_mgr.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid11.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid14.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid19.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid22.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid28.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid2E.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid2F.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid31.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid34.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid36.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid37.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid38.h"
#include "diag/diag_server/include/uds/diag_server_uds_sid85.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerUdsMgr* DiagServerUdsMgr::instance_ = nullptr;
std::mutex DiagServerUdsMgr::mtx_;

DiagServerUdsMgr::DiagServerUdsMgr()
{
}

DiagServerUdsMgr*
DiagServerUdsMgr::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerUdsMgr();
        }
    }

    return instance_;
}

void
DiagServerUdsMgr::Init()
{
    DG_DEBUG << "DiagServerUdsMgr::Init";
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_BASE, new DiagServerUdsBase()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_ECU_RESET, new DiagServerUdsSid11()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_DTC_CLEAR, new DiagServerUdsSid14()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DTC_INFO, new DiagServerUdsSid19()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_READ_DATA_IDENTIFIER, new DiagServerUdsSid22()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_COMMUNICATION_CONTROL, new DiagServerUdsSid28()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_WRITE_DATA_IDENTIFIER, new DiagServerUdsSid2E()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_INPUT_OUTPUT_CONTROL, new DiagServerUdsSid2F()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_ROUTINE_CONTROL, new DiagServerUdsSid31()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_DOWNLOAD, new DiagServerUdsSid34()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_TRANSFER_DATA, new DiagServerUdsSid36()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_TRANSFER_EXIT, new DiagServerUdsSid37()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_REQUEST_FILE_TRANSFER, new DiagServerUdsSid38()));
    sid_base_map_.insert(std::make_pair(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_CONTROL_DTC_SET, new DiagServerUdsSid85()));
}

void
DiagServerUdsMgr::DeInit()
{
    DG_DEBUG << "DiagServerUdsMgr::DeInit";
    for (auto& item : sid_base_map_) {
        if (item.second != nullptr) {
            delete item.second;
            item.second = nullptr;
        }
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
DiagServerUdsMgr::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerUdsMgr::AnalyzeUdsMessage udsdata.size: " << udsMessage.udsData.size()
                                                                     <<  " sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                                     <<  " ta: " << UINT16_TO_STRING(udsMessage.udsTa);
    if (udsMessage.udsData.empty()) {
        DG_DEBUG << "DiagServerUdsMgr::AnalyzeUdsMessage udsData is empty!";
        return;
    }

    uint8_t sid = udsMessage.udsData[0];
    DG_DEBUG << "DiagServerUdsMgr::AnalyzeUdsMessage sid: " << UINT8_TO_STRING(sid);

    auto itr_sid_find = sid_base_map_.find(sid);
    if (itr_sid_find != sid_base_map_.end()) {
        itr_sid_find->second->AnalyzeUdsMessage(udsMessage);
    }
}

void
DiagServerUdsMgr::sendNegativeResponse(DiagServerServiceRequestOpc eOpc, DiagServerUdsMessage& udsMessage)
{
    if (NULL == sid_base_map_[eOpc]) {
        DG_DEBUG << "DiagServerUdsMgr::sendNegativeResponse, invalid opc:" << eOpc;
        return;
    }

    sid_base_map_[eOpc]->NegativeResponse(udsMessage);
}

void
DiagServerUdsMgr::sendPositiveResponse(DiagServerServiceRequestOpc eOpc, DiagServerUdsMessage& udsMessage)
{
    if (NULL == sid_base_map_[eOpc]) {
        DG_DEBUG << "DiagServerUdsMgr::sendPositiveResponse, invalid opc:" << eOpc;
        return;
    }

    sid_base_map_[eOpc]->PositiveResponse(udsMessage);
}

DiagServerUdsBase*
DiagServerUdsMgr::getSidService(DiagServerServiceRequestOpc eOpc)
{
    if (NULL == sid_base_map_[eOpc]) {
        DG_DEBUG << "DiagServerUdsMgr::sendPositiveResponse, invalid opc:" << eOpc;
        return NULL;
    }

    return sid_base_map_[eOpc];
}


}  // namespace diag
}  // namespace netaos
}  // namespace hozon
