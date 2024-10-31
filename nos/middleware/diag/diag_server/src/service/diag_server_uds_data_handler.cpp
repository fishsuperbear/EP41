#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/transport/diag_server_transport.h"
#include "diag/diag_server/include/service/diag_server_uds_data_handler.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

DiagServerUdsDataHandler* DiagServerUdsDataHandler::instance_ = nullptr;
std::mutex DiagServerUdsDataHandler::mtx_;

DiagServerUdsDataHandler*
DiagServerUdsDataHandler::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerUdsDataHandler();
        }
    }

    return instance_;
}

DiagServerUdsDataHandler::DiagServerUdsDataHandler()
{
}

void
DiagServerUdsDataHandler::Init()
{
    DG_DEBUG << "DiagServerUdsDataHandler::Init";
}

void
DiagServerUdsDataHandler::DeInit()
{
    DG_DEBUG << "DiagServerUdsDataHandler::DeInit";

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
DiagServerUdsDataHandler::RecvUdsMessage(DiagServerUdsMessage& udsMessage)
{
    if (udsMessage.udsData.empty()) {
        DG_ERROR << "DiagServerUdsDataHandler::RecvUdsMessage udsMessage.udsData is empty!";
        return;
    }

    DG_INFO << "DiagServerUdsDataHandler::RecvUdsMessage sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                    << " ta: " << UINT16_TO_STRING(udsMessage.udsTa)
                                                    << " taType: " << udsMessage.taType
                                                    << " udsdata.size: " << udsMessage.udsData.size()
                                                    << " udsdata: " << UINT8_VEC_TO_STRING(udsMessage.udsData);
    udsMessage.suppressPosRspMsgIndBit = 0x00;
    udsMessage.pendingRsp = 0x00;

    DiagServerSessionHandler::getInstance()->RecvUdsMessage(udsMessage);
}

void
DiagServerUdsDataHandler::ReplyUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_INFO << "DiagServerUdsDataHandler::ReplyUdsMessage sa: " << UINT16_TO_STRING(udsMessage.udsSa)
                                                     << " ta: " << UINT16_TO_STRING(udsMessage.udsTa)
                                                     << " taType: " << udsMessage.taType
                                                     << " udsdata.size: " << udsMessage.udsData.size()
                                                     << " udsdata: " << UINT8_VEC_TO_STRING(udsMessage.udsData);
    if (DiagTargetAddressType::kFunctional == udsMessage.taType) {
        if (DiagServerNrcErrc::kNegativeHead == udsMessage.udsData[0]) {
            switch (udsMessage.udsData[udsMessage.udsData.size() - 1])
            {
                case DiagServerNrcErrc::kServiceNotSupported:
                    return;
                case DiagServerNrcErrc::kSubfunctionNotSupported:
                    return;
                case DiagServerNrcErrc::kRequestOutOfRange:
                    return;
                case DiagServerNrcErrc::kSubFunctionNotSupportedInActiveSession:
                    return;
                case DiagServerNrcErrc::kServiceNotSupportedInActiveSession:
                    return;
                default:
                    break;
            }
        }
    }

    DiagServerTransport::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerUdsDataHandler::NotifyMessageFailure(const DiagServerUdsMessage& udsMessage)
{

}

void
DiagServerUdsDataHandler::TransmitConfirmation(const DiagServerUdsMessage& udsMessage, const bool confirmResult)
{

}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
