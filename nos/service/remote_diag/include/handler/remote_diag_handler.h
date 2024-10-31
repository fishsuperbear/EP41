
#ifndef REMOTE_DIAG_DATA_HANDLER_H
#define REMOTE_DIAG_DATA_HANDLER_H

#include <mutex>
#include <iostream>
#include <unordered_map>

#include "json/json.h"
#include "remote_diag/include/transport/remote_diag_doip_dispatcher.h"
#include "remote_diag/include/transport/remote_diag_rocketmq_dispatcher.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

class RemoteDiagHandler {

public:
    static RemoteDiagHandler* getInstance();

    void Init();
    void DeInit();

    void RecvRemoteMessage(const Json::Value& message);
    void ReplyRemoteMessage(const Json::Value& message);

    void SetCurrExtension(REMOTE_DIAG_EXTENSION extension) {curr_extension_ = extension;}
    REMOTE_DIAG_EXTENSION GetCurrExtension() {return curr_extension_;}

    void SetRemoteRequestStatus(const uint16_t address, bool status);
    void SetVehicleSpeed(const double vehicleSpeed) {vehicle_speed_ = vehicleSpeed;}

    void FileDownloadCompleteCallback(const std::string& filePath);

    static std::vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = REGEX);

private:
    RemoteDiagHandler();
    RemoteDiagHandler(const RemoteDiagHandler &);
    RemoteDiagHandler & operator = (const RemoteDiagHandler &);

private:
    bool IsExecutableRequest(const uint16_t sa, const uint16_t ta, DiagUdsBusType& busType, DiagUdsNrcErrc& nrc);

    void ConnectConfirmReponse(const Json::Value& message);
    void UdsCommandReuqest(const Json::Value& message);
    void FileUploadReuqest(const Json::Value& message);
    void FileDownloadReuqest(const Json::Value& message);
    void PluginRunReuqest(const Json::Value& message);
    void PluginRunResultReuqest(const Json::Value& message);
    void SwitchControlReuqest(const Json::Value& message);
    void QueryDirInfoReuqest(const Json::Value& message);

    static std::string PrintJsonData(const Json::Value& value);
    static std::string NegativeReponseStr(const DiagUdsNrcErrc& nrc);

private:
    static RemoteDiagHandler* instance_;
    static std::mutex mtx_;

    RemoteDiagDoipDispatcher* doip_dispatcher_;
    RemoteDiagRocketMQDispatcher* rocketmq_dispatcher_;

    REMOTE_DIAG_EXTENSION curr_extension_;

    // unordered_map<remoteaddress, requeststatus>
    std::unordered_map<uint16_t, bool> remote_diag_request_status_info_;

    double vehicle_speed_;
};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // #define REMOTE_DIAG_DATA_HANDLER_H
