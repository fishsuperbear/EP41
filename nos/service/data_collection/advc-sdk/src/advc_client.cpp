#include "advc_client.h"

#include <memory>
#include <mutex>
#include <utility>

#include "Poco/Net/HTTPSStreamFactory.h"
#include "Poco/Net/HTTPStreamFactory.h"
#include "op/service_op.h"
#include "util/default_time_util.h"

namespace advc {

bool AdvcClient::s_init = false;
bool AdvcClient::s_poco_init = false;
int AdvcClient::s_advc_obj_num = 0;
std::mutex g_init_lock;
std::shared_ptr<Time> Time::time_util;

AdvcClient::AdvcClient() {
    Time::time_util = std::make_shared<DefaultTimeUtil>();
    AdvcInit();
}

AdvcClient::AdvcClient(std::shared_ptr<Time> time_ptr) {
    Time::time_util = std::move(time_ptr);
    AdvcInit();
}

AdvcClient::~AdvcClient() { AdvcUInit(); }

int AdvcClient::AdvcInit() {
    std::lock_guard<std::mutex> lock(g_init_lock);
    ++s_advc_obj_num;
    if (!s_init) {
        if (!s_poco_init) {
            Poco::Net::HTTPStreamFactory::registerFactory();
            Poco::Net::HTTPSStreamFactory::registerFactory();
            Poco::Net::initializeSSL();
            s_poco_init = true;
        }
        s_init = true;
    }
    return 0;
}

GetAesKeyResp AdvcClient::GetAesKey(std::string &uploadToken) {
    ServiceOp serviceOp;
    UploadTokenStruct uploadTokenStruct{};
    serviceOp.DecodeUploadToken(CodecUtil::Base64Decode(uploadToken), uploadTokenStruct);
    GetAesKeyReq getAesKeyReq{};
    {
        getAesKeyReq.customVehicleId = uploadTokenStruct.customVehicleId;
    }
    try {
        GetAesKeyResp getAesKeyResp;
        serviceOp.getAesKey(uploadTokenStruct, &getAesKeyResp);
        return getAesKeyResp;
    } catch (const std::exception &ex) {
        GetAesKeyResp getAesKeyResp{};
        {
            getAesKeyResp.responseMetadata.error.message = "Exception:" + std::string(ex.what());
        }
        return getAesKeyResp;
    }
}

UploadFileResp AdvcClient::UploadFile(UploadFileReq &req) {
    try {
        std::atomic<bool> interrupt{};
        UploadFileResp uploadFileResp = m_service_op.UploadFile(req, interrupt);
        return uploadFileResp;
    } catch (const std::exception &ex) {
        UploadFileResp uploadFileResp = UploadFileResp(false, ErrorObj("Exception:" + std::string(ex.what())));
        return uploadFileResp;
    }
}

UploadFileResp AdvcClient::UploadFile(UploadFileReq &req, std::atomic<bool> &interrupt) {
    try {
        UploadFileResp uploadFileResp = m_service_op.UploadFile(req, interrupt);
        return uploadFileResp;
    } catch (const std::exception &ex) {
        UploadFileResp uploadFileResp = UploadFileResp(false, ErrorObj("Exception:" + std::string(ex.what())));
        return uploadFileResp;
    }
}

void AdvcClient::AdvcUInit() {
    std::lock_guard<std::mutex> lock(g_init_lock);
    --s_advc_obj_num;
    if (s_init && s_advc_obj_num == 0) {
        s_init = false;
    }
}

UploadFileResp
AdvcClient::ResumableUploadFile(UploadFileReq &req, std::string &checkpointDirPath) {
    try {
        std::atomic<bool> interrupt{};
        UploadFileResp uploadFileResp = m_service_op.ResumableUploadFile(req, checkpointDirPath, interrupt);
        return uploadFileResp;
    } catch (const std::exception &ex) {
        UploadFileResp uploadFileResp = UploadFileResp(false, ErrorObj("Exception:" + std::string(ex.what())));
        return uploadFileResp;
    }
}

UploadFileResp
AdvcClient::ResumableUploadFile(UploadFileReq &req, std::string &checkpointDirPath, std::atomic<bool> &interrupt) {
    try {
        UploadFileResp uploadFileResp = m_service_op.ResumableUploadFile(req, checkpointDirPath, interrupt);
        return uploadFileResp;
    } catch (const std::exception &ex) {
        UploadFileResp uploadFileResp = UploadFileResp(false, ErrorObj("Exception:" + std::string(ex.what())));
        return uploadFileResp;
    }
}
}  // namespace advc
