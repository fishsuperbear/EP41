/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_client_impl.cpp
 * @Date: 2023/10/27
 * @Author: cheng
 * @Desc: --
 */

#include "client/include/dc_client_impl.h"

#include <vector>
#include <fstream>
#include <sstream>

#include "utils/include/path_utils.h"
#include "utils/include/trans_utils.h"
#include "cm/include/method.h"
#include "idl/generated/data_collection_info.h"
#include "idl/generated/data_collection_infoPubSubTypes.h"
#include "utils/include/dc_logger.hpp"
#include "utils/include/time_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

DcClientImpl::DcClientImpl() : client_(std::make_shared<triggerInfoPubSubType>(), std::make_shared<triggerResultPubSubType>()),
                               clientUpload_(std::make_shared<triggerUploadInfoPubSubType>(), std::make_shared<triggerResultPubSubType>()) {

    cout << "DcClientImpl" << endl;
}

DcResultCode DcClientImpl::Init(const std::string client_name, const uint32_t max_wait_millis) {
    if (isInit_.load(std::memory_order::memory_order_acquire)) {
        return DcResultCode::DC_UNSUPPORT;
    }
    std::lock_guard<std::mutex> lg(mtx_);
    isInit_.store(true, std::memory_order::memory_order_release);
    client_.Init(0, "serviceDataCollection");
    clientUpload_.Init(0, "serviceDataCollectionUpload");
    maxWaitTimeMs_ = max_wait_millis;
    clientName_ = client_name;
    return DC_OK;
}

DcResultCode DcClientImpl::DeInit() {
    if (!isInit_.load(std::memory_order::memory_order_acquire)) {
        return DcResultCode::DC_UNSUPPORT;
    }
    std::lock_guard<std::mutex> lg(mtx_);
    client_.Deinit();
    clientUpload_.Deinit();
    isInit_.store(false, std::memory_order::memory_order_release);
    return DC_OK;
}

DcResultCode DcClientImpl::CollectTrigger(uint32_t trigger_id) {
    return CollectTriggerDesc(trigger_id, 0);
};

DcResultCode DcClientImpl::CollectTriggerDesc(uint32_t trigger_id, uint64_t time){
    if (!isInit_.load(std::memory_order::memory_order_acquire)) {
        return DcResultCode::DC_UNSUPPORT;
    }
    std::lock_guard<std::mutex> lg(mtx_);
    if (trigger_id < 100 || (trigger_id > 10000 && trigger_id<=20000)) {
        return DcResultCode::DC_INVALID_PARAM;
    }
    int online;
    int maxRetryTimes = (int)maxWaitTimeMs_ / 500;
    if (maxRetryTimes < 1) {
        online = client_.WaitServiceOnline(maxWaitTimeMs_);
        if (online != 0) {
            return DcResultCode::DC_SERVICE_NO_READY;
        }
    }
    for (int i = 1; i <= maxRetryTimes; i++) {
        online = client_.WaitServiceOnline(500 + (i / maxRetryTimes) * (maxWaitTimeMs_ % 500));
        if (online == 0) {
            cout << "retry times:" << i << endl;
            break;
        }
        if (i == maxRetryTimes) {
            cout << "service is not online after " << maxRetryTimes << " times try;" << endl;
            return DcResultCode::DC_TIMEOUT;
        }
    }
    cout << "Service online result:" << online << endl;
    std::shared_ptr<triggerInfo> req_data = std::make_shared<triggerInfo>();
    req_data->clientName(clientName_);
    if (time == 0) {
        req_data->type("trigger");
        req_data->value(std::to_string(trigger_id));  // "emergencyBraking"
    } else {
        req_data->type("triggerdesc");
        req_data->value(std::to_string(trigger_id)+"|"+TimeUtils::formatTimeStrForFileName(time));
    }
    std::shared_ptr<triggerResult> res_data = std::make_shared<triggerResult>();
    res_data->retCode(1);
    res_data->msg("Service not online");
    auto res = client_.Request(req_data, res_data, 5000 * 5);
    if (res != 0) {
        return DcResultCode::DC_INNER_ERROR;
    }
    cout << "result:" << res_data->msg() << endl;
    return DC_OK;
}

DcResultCode DcClientImpl::Upload(std::vector<std::string> &path_list, std::string file_type, std::string file_name, uint16_t cache_file_num) {
    if (!isInit_.load(std::memory_order::memory_order_acquire)) {
        return DcResultCode::DC_UNSUPPORT;
    }

    if ((path_list.empty()) || (path_list.size() > 200)) {
        return DcResultCode::DC_INVALID_PARAM;
    }

    std::vector<std::string> all_file_type_vec = {"CAN","TRIGGER","FAULT","ETH","CALIBRATION","LOG","PLANNING","MCU"};
    if (std::find(all_file_type_vec.begin(), all_file_type_vec.end(), file_type) == all_file_type_vec.end()) {
        return DcResultCode::DC_INVALID_PARAM;
    }

    if ((cache_file_num < 0) || (cache_file_num > 20)) {
        return DcResultCode::DC_INVALID_PARAM;
    }

    if (file_name.empty()) {
        std::string file_type_lower;
        for (char c_up : file_type) {
            char c_low = std::tolower(c_up);
            file_type_lower = file_type_lower + c_low;
        }
        file_name = "EP41_ORIN_" + file_type_lower + "-%Y%m%d-%H%M%S.tar.gz";
    } else if ((file_name.find(".tar.gz")) != (file_name.size() - 7)) {
        return DcResultCode::DC_INVALID_PARAM;
    }

    std::lock_guard<std::mutex> lg(mtx_);
    int online;
    int maxRetryTimes = (int)maxWaitTimeMs_ / 500;
    if (maxRetryTimes < 1) {
        online = clientUpload_.WaitServiceOnline(maxWaitTimeMs_);
        if (online != 0) {
            return DcResultCode::DC_SERVICE_NO_READY;
        }
    }
    for (int i = 1; i <= maxRetryTimes; i++) {
        online = clientUpload_.WaitServiceOnline(500 + (i / maxRetryTimes) * (maxWaitTimeMs_ % 500));
        if (online == 0) {
            cout << "retry times:" << i << endl;
            break;
        }
        if (i == maxRetryTimes) {
            cout << "service is not online after " << maxRetryTimes << " times try;" << endl;
            return DcResultCode::DC_TIMEOUT;
        }
    }
    cout << "Service online result:" << online << endl;
    std::shared_ptr<triggerUploadInfo> req_data = std::make_shared<triggerUploadInfo>();
    req_data->clientName(clientName_);
    req_data->type("trigger_upload");
    req_data->pathList(path_list);
    req_data->fileType(file_type);
    req_data->fileName(file_name);
    req_data->cacheFileNum(cache_file_num);
    req_data->deleteAfterCompress(deleteAfterCompress.load(std::memory_order::memory_order_acquire));
    std::shared_ptr<triggerResult> res_data = std::make_shared<triggerResult>();
    res_data->retCode(1);
    res_data->msg("Service not online");
    auto res = clientUpload_.Request(req_data, res_data, 5000 * 5);
    if (res != 0) {
        return DcResultCode::DC_INNER_ERROR;
    }
    cout << "result:" << res_data->msg() << endl;
    deleteAfterCompress.store(false, std::memory_order::memory_order_release);
    return DC_OK;
}

DcResultCode DcClientImpl::CollectTriggerDesc(uint32_t trigger_id, uint64_t time, std::string desc){
    std::string fileName = "EP41_ORIN_trigger-desc-"+TimeUtils::formatTimeStrForFileName(time)+std::to_string(trigger_id)+".tar.gz";
    std::vector<char> descVec(desc.begin(),desc.end());
    return Upload(descVec, "TRIGGER",fileName, 5);
}

DcResultCode DcClientImpl::Upload(std::vector<char> &data, std::string file_type, std::string file_name, uint16_t cache_file_num) {
    if (!isInit_.load(std::memory_order::memory_order_acquire)) {
        return DcResultCode::DC_UNSUPPORT;
    }
    if (file_name.empty()) {
        std::string file_type_lower;
        for (char c_up : file_type) {
            char c_low = std::tolower(c_up);
            file_type_lower = file_type_lower + c_low;
        }
        file_name = TransUtils::stringTransFileName("EP41_ORIN_" + file_type_lower + "-%Y%m%d-%H%M%S.tar.gz");
    } else if ((file_name.find(".tar.gz")) != (file_name.size() - 7)) {
        return DcResultCode::DC_INVALID_PARAM;
    }
    std::string file_name_prefix = file_name.substr(0, file_name.find(".tar.gz"));
    std::string folder = TransUtils::stringTransFileName("/opt/usr/col/toupload/%Y%m%d-%H%M%S/");
    PathUtils::createFoldersIfNotExists(folder);
    std::ofstream output_file(PathUtils::getFilePath(folder, file_name_prefix), std::ios::out | std::ios::binary);
    output_file.write(data.data(), data.size());
    output_file.close();
    std::vector<std::string> path_list;
    path_list.push_back(PathUtils::getFilePath(folder, file_name_prefix));
    deleteAfterCompress.store(true, std::memory_order::memory_order_release);
    return Upload(path_list, file_type, file_name, cache_file_num);
}
}  // namespace dc
}  // namespace netaos
}  // namespace hozon
