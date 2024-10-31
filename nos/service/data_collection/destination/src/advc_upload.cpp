/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: advc_upload.cpp
 * @Date: 2023/08/17
 * @Author: cheng
 * @Desc: --
 */

#include "destination/include/advc_upload.h"
#include <sys/stat.h>
#include <unistd.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>
#include "Poco/Exception.h"
#include "advc_sys_config.h"
#include "config_param.h"
#include "encrypt/aes_cbc.h"
#include "encrypt/aes_gcm.h"
#include "tsp_comm.h"
#include "utils/include/custom_time_util.h"
#include "utils/include/dc_logger.hpp"
#include "utils/include/path_utils.h"
#include "utils/include/time_utils.h"

namespace hozon {
namespace netaos {
namespace dc {

using namespace hozon::netaos::https;
using namespace hozon::netaos::cfg;

static const std::string TOUPLOAD_DIR = "/opt/usr/col/toupload/";
static const int GET_UPLOAD_TOKEN_RETRY_COUNTS = 3;

std::string AdvcUploadTask::m_encrpytion_key;
std::string AdvcUploadTask::m_encryption_version_id;
std::mutex AdvcUploadTask::m_encryption_key_mutex;
std::string AdvcUploadTask::m_raw_key;
std::atomic_bool AdvcUploadTask::selfActive_;

AdvcUploadTask::AdvcUploadTask() {
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask";
    m_status = std::make_shared<std::atomic<TaskStatus>>(TaskStatus::INITIAL);
    m_upload_files = std::make_shared<std::vector<std::string>>();
    m_origin_path = std::make_shared<std::vector<std::string>>();
    m_protocol_id = std::make_shared<std::string>();
    m_upload_type = std::make_shared<std::string>();
    m_toupload_dir = std::make_shared<std::string>();
    m_checkpoint = std::make_shared<std::string>();
    m_stop_flag = std::make_shared<std::atomic_bool>(false);
}

AdvcUploadTask::~AdvcUploadTask() {
    DC_SERVER_LOG_DEBUG << "~AdvcUploadTask";
}

void AdvcUploadTask::configure(std::string type, YAML::Node& node) {
    DC_SERVER_LOG_DEBUG << "configure YAML";
    if (type == "uploadType") {
        if (!node["uploadType"]) {
            DC_SERVER_LOG_ERROR << "AdvcUploadTask: uploadType missed";
            set_status(TaskStatus::ERROR);
            return;
        }
        *m_upload_type = node["uploadType"].as<std::string>();
        set_status(TaskStatus::CONFIGURED);
        return;
    }
    auto uploadFiles = node.as<Configuration_>();
    if (uploadFiles.protocolId.empty()) {
        DC_SERVER_LOG_ERROR << "AdvcUploadTask: protocolId missed";
        set_status(TaskStatus::ERROR);
    } else {
        *m_protocol_id = uploadFiles.protocolId;
        DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_protocol_id: " << *m_protocol_id;
    }
    if (uploadFiles.uploadType.empty()) {
        if ((*m_upload_type).empty()) {
            DC_SERVER_LOG_ERROR << "AdvcUploadTask: uploadType missed";
            set_status(TaskStatus::ERROR);
            return;
        }
    } else {
        *m_upload_type = uploadFiles.uploadType;
        DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_upload_type: " << *m_upload_type;
    }
    *m_checkpoint = uploadFiles.checkPoint;
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_checkpoint: " << *m_checkpoint;
    if (uploadFiles.retryCount <= 0) {
        m_retry_count = Configuration_::DEF_RETRY_COUNT;
    } else {
        m_retry_count = uploadFiles.retryCount;
    }
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_retry_count: " << m_retry_count;
    if (uploadFiles.retryInterval < 0) {
        m_retry_interval = Configuration_::DEF_RETRY_INTERVAL;
    } else {
        m_retry_interval = uploadFiles.retryInterval;
    }
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_retry_interval: " << m_retry_interval;
    if (uploadFiles.enableEncrypt) {
        m_enable_encrypt = true;
    }
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_enable_encrypt: " << m_enable_encrypt;
    if (uploadFiles.deleteAfterUpload) {
        m_delete_after_upload = true;
    }
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_delete_after_upload: " << m_delete_after_upload;
}

void AdvcUploadTask::configure(std::string type, DataTrans& node) {
    DC_SERVER_LOG_DEBUG << "configure DataTrans";
    for (auto type2PathList : node.pathsList) {
        for (auto& path : type2PathList.second) {
            m_origin_path->emplace_back(path);
        }
    }
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask: m_origin_path size: " << m_origin_path->size();
    if (get_status() == TaskStatus::INITIAL) {
        set_status(TaskStatus::CONFIGURED);
    }
    DC_SERVER_LOG_DEBUG << "AdvcUploadTask: configure DataTrans\n";
}

void AdvcUploadTask::active() {
    if (m_origin_path->empty()) {
        DC_SERVER_LOG_WARN << "AdvcUploadTask: nothing to upload";
        set_status(TaskStatus::FINISHED);
        return;
    }
    DC_SERVER_LOG_DEBUG << "active";
    if (get_stop_flag()){
        return;
    }
    selfActive_ = true;
    if (m_threadpool->getQueueSize() > 10) {
        DC_SERVER_LOG_ERROR << "the queue size is above 10";
        this->set_status(TaskStatus::ERROR);
        return;
    }
    if (get_status() != TaskStatus::CONFIGURED) {
        DC_SERVER_LOG_ERROR << "the advc upload task haven't right configuration";
        this->set_status(TaskStatus::ERROR);
        return;
    }
    if (m_delete_after_upload || m_enable_encrypt) {
        *m_toupload_dir =
            TOUPLOAD_DIR + *m_upload_type + "/" + TimeUtils::formatTimeStrForFileName(TimeUtils::getDataTimestamp());
        PathUtils::createFoldersIfNotExists(*m_toupload_dir);
        DC_SERVER_LOG_DEBUG << "AdvcUploadTask: create fault dir " << *m_toupload_dir;
    }

    if (m_delete_after_upload) {
        std::vector<std::string> upload_files;
        for (auto& upload_file : *m_origin_path) {
            auto toupload_file = *m_toupload_dir + "/" + PathUtils::getFileName(upload_file);
            if (!PathUtils::renameFile(upload_file, toupload_file)) {
                DC_SERVER_LOG_INFO << "AdvcUploadTask: move upload file " << upload_file << " to " << toupload_file
                                   << " error";
            } else {
                DC_SERVER_LOG_DEBUG << "AdvcUploadTask: move upload file " << toupload_file;
                upload_files.emplace_back(toupload_file);
            }
        }
        m_origin_path->swap(upload_files);
    }
    for (auto path : *m_origin_path) {
        if (PathUtils::isFileExist(path)) {
            m_upload_files->emplace_back(path);
        } else if (PathUtils::isDirExist(path)) {
            PathUtils::getFiles(path, *m_upload_files);
        }
    }
    m_threadpool->add([this]() {  // in different thread
        auto uploadTask = *this;
        pthread_setname_np(pthread_self(), "advc_upload");
        DC_SERVER_LOG_INFO << "AdvcUploadTask: start executing uploadFiles";
        auto res = uploadTask.uploadFiles();
        if (!res) {
            uploadTask.set_status(TaskStatus::ERROR);
            DC_SERVER_LOG_ERROR << "AdvcUploadTask: uploadFiles error";
            return;
        }
        if (m_delete_after_upload || m_enable_encrypt) {
            PathUtils::removeFolder(*m_toupload_dir);
            DC_SERVER_LOG_DEBUG << "AdvcUploadTask: delete old dir：" << *m_toupload_dir;
        }
        uploadTask.set_status(TaskStatus::FINISHED);
        DC_SERVER_LOG_INFO << "AdvcUploadTask: finish executing uploadFiles";
    });
    this->set_status(TaskStatus::RUNNING);
    selfActive_ = false;
}

void AdvcUploadTask::deactive() {
    DC_SERVER_LOG_DEBUG << "deactive";
    set_stop_flag(true);
    for (int i = 0; i < 10; i++) {
        if (!selfActive_) {
            set_status(TaskStatus::ERROR);
            break;
        }
        TimeUtils::sleep(500);
    }
    if (!selfActive_) {
        DC_SERVER_LOG_INFO << "AdvcUploadTask: deactivated";
    } else {
        DC_SERVER_LOG_ERROR << "AdvcUploadTask: can't deactivated";
    }
}

TaskStatus AdvcUploadTask::getStatus() {
    return m_status->load(std::memory_order_acquire);
}

std::string AdvcUploadTask::get_upload_token() {
    auto cfgMgr = ConfigParam::Instance();
    cfgMgr->Init();
    std::string upload_token;
    auto& https_req = TspComm::GetInstance();
    https_req.Init();
    if (get_stop_flag()) {
        DC_SERVER_LOG_INFO << "AdvcUploadTask: stopped";
        return upload_token;
    }
    for (int i = 0; i < GET_UPLOAD_TOKEN_RETRY_COUNTS; i++) {
        std::future<TspComm::TspResponse> future_ret = https_req.RequestUploadToken();
        TspComm::TspResponse req_ret = future_ret.get();
        DC_SERVER_LOG_INFO << "AdvcUploadTask invokes RequestUploadToken result_code:" << req_ret.result_code
                           << " response:" << req_ret.response;
        upload_token = req_ret.response;
        if (!upload_token.empty() || get_stop_flag()) {
            break;
        }
    }
    if (upload_token.empty()) {
        DC_SERVER_LOG_ERROR << "AdvcUploadTask: cannot get upload token";
        set_status(TaskStatus::ERROR);
        return upload_token;
    } else {
        DC_SERVER_LOG_DEBUG << "AdvcUploadTask: upload token: " << upload_token;
    }
    return upload_token;
}

bool AdvcUploadTask::uploadFiles() {

    DC_SERVER_LOG_DEBUG << "uploadFiles";
    std::string upload_token = get_upload_token();
    if (upload_token.empty()) {
        return false;
    }
    if (get_stop_flag()) {
        DC_LOG_INFO << "AdvcUploadTask: stopped";
        return true;
    }
    for (auto upload_file : *m_upload_files) {
        try {
            if (get_stop_flag()) {
                DC_SERVER_LOG_INFO << "AdvcUploadTask is stopped";
                return true;
            }
            // 单文件上传
            auto ctu = std::make_shared<CustomTimeUtil>();
            advc::AdvcClient advc_client(ctu);
            TaskResult res = uploadSingleFile(advc_client, upload_file, upload_token, m_stop_flag);
            if (res != TaskResult::OK) {
                ++error_count;
            } else {
                if (m_delete_after_upload) {
                    if (!PathUtils::removeFile(upload_file)) {
                        DC_SERVER_LOG_ERROR << "AdvcUploadTask: remove uploaded file " << upload_file << " error";
                    } else {
                        DC_SERVER_LOG_INFO << "AdvcUploadTask: remove uploaded file " << upload_file;
                    }
                }
            }
        } catch (const Poco::Exception& e) {
            DC_SERVER_LOG_ERROR << "AdvcUploadTask catch Poco exception: " << e.message();
            ++error_count;
        } catch (const std::exception& e) {
            DC_SERVER_LOG_ERROR << "AdvcUploadTask catch exception: " << e.what();
            ++error_count;
        }
    }
    DC_SERVER_LOG_INFO << "AdvcUploadTask is completed";
    return error_count == 0;
}

AdvcUploadTask::TaskResult AdvcUploadTask::requestEncryptKey(std::string& upload_token, std::string& encrpytion_key,
                                                             std::string& version_id, std::string& raw_key) {
    {
        std::lock_guard<std::mutex> lock(m_encryption_key_mutex);
        if (!m_encrpytion_key.empty()) {
            encrpytion_key = m_encrpytion_key;
            version_id = m_encryption_version_id;
            raw_key = m_raw_key;
            return TaskResult::OK;
        }
    }
    auto ctu = std::make_shared<CustomTimeUtil>();
    advc::AdvcClient advc_client(ctu);
    advc::GetAesKeyResp res = advc_client.GetAesKey(upload_token);
    if (res.success && !res.Result.plainText.empty()) {
        encrpytion_key = res.Result.plainText;
        version_id = res.Result.versionId;
        raw_key = advc::CodecUtil::Base64Decode(res.Result.plainText);
        {
            std::lock_guard<std::mutex> lock(m_encryption_key_mutex);
            m_encrpytion_key = encrpytion_key;
            m_encryption_version_id = version_id;
            m_raw_key = raw_key;
        }
        // Note: do not log the key content for security.
        DC_SERVER_LOG_INFO << "AdvcUploadTask: encryption key(base64) size: " << res.Result.plainText.size()
                           << ", version_id: " << res.Result.versionId << ", raw_key size: " << raw_key.size() * 8;
        return TaskResult::OK;
    }
    DC_SERVER_LOG_ERROR << "AdvcUploadTask: get encryption key error, error.codeN: " << res.responseMetadata.error.codeN
                        << ", error: " << res.responseMetadata.error.ToJsonString();
    if (res.responseMetadata.error.codeN == ADVC_RESP_CODE_TOKEN_EXPIRED) {
        upload_token = get_upload_token();
    }
    return TaskResult::REQUEST_ENCRYPTKEY_FAILED;
}

std::string AdvcUploadTask::encryptData(std::string encryption_key, std::string raw_key, std::string upload_file) {
    std::string encrypted_data;
    std::ifstream ifs(upload_file, std::ifstream::binary);
    if (ifs) {
        std::istreambuf_iterator<char> begin(ifs), end;
        std::string raw_data(begin, end);
        encrypted_data = advc::AesGcmEncryptor::AesGcmEncryptString(raw_data, encryption_key, raw_key.size() * 8);
        DC_SERVER_LOG_DEBUG << "AdvcUploadTask: enrypt data completed, raw data size: " << raw_data.size()
                            << ", encrypted data size: " << encrypted_data.size();
    } else {
        DC_SERVER_LOG_ERROR << "AdvcUploadTask: " << upload_file << " open error";
    }
    return encrypted_data;
}

static uint32_t getTimeSpecificSec() {
    struct timespec time = {0};
    auto cfgMgr = ConfigParam::Instance();
    cfgMgr->Init();
    int64_t value;
    auto res = cfgMgr->GetParam<int64_t>("time/mp_offset", value);
    if (CfgResultCode::CONFIG_OK != res || value == 0) {
        DC_SERVER_LOG_ERROR<<"get cfg result is "<<res;
        clock_gettime(CLOCK_VIRTUAL, &time);
        return static_cast<uint32_t>(time.tv_sec);
    } else {
        clock_gettime(CLOCK_REALTIME, &time);
        return static_cast<uint32_t>(time.tv_sec) + value;
    }
    //    if (GET_TIMER_USE_PLANE == "DP") {
    //        clock_gettime(CLOCK_REALTIME, &time);
    //    } else {
    //        clock_gettime(CLOCK_VIRTUAL, &time);
    //    }
    //    return static_cast<uint32_t>(time.tv_sec) + TIME_OFFSET_SECOND;
}

AdvcUploadTask::TaskResult AdvcUploadTask::uploadSingleFile(advc::AdvcClient& advc_client, std::string upload_file,
                                                            std::string upload_token,
                                                            std::shared_ptr<std::atomic_bool> stop_flag) {
    DC_SERVER_LOG_DEBUG << "uploadSingleFile";
    for (int i = 0; i < m_retry_count; ++i) {
        // Get encryption key and encrypt upload data if need encryption.
        std::string upload_data;
        advc::EncryptMeta encrypt_meta{};
        std::ifstream ifs;
        std::istringstream iss;
        uint64_t file_size = 0;
        if (m_enable_encrypt) {
            // Get encrpytion key from advc server.
            std::string temp_encryption_key;
            std::string temp_version_id;
            std::string temp_raw_key;
            requestEncryptKey(upload_token, temp_encryption_key, temp_version_id, temp_raw_key);
            if (temp_encryption_key.empty()) {
                continue;
            }
            // Encrypt upload data.
            {
                upload_data = encryptData(temp_encryption_key, temp_raw_key, upload_file);
                if (upload_data.empty()) {
                    DC_SERVER_LOG_ERROR << "AdvcUploadTask: encrypt data error";
                    continue;
                }
                upload_file += ".encrypted";
                upload_file = *m_toupload_dir + "/" + PathUtils::getFileName(upload_file);
                std::ofstream out_encrypt_file(upload_file, std::ios::binary);
                if (out_encrypt_file) {
                    out_encrypt_file.write(upload_data.c_str(), upload_data.size());
                    out_encrypt_file.flush();
                }
                file_size = upload_data.size();
                iss = std::istringstream(std::move(upload_data));
                encrypt_meta.AlgorithmMode = "GCM";
                encrypt_meta.KeyVersionId = temp_version_id;
                encrypt_meta.EncryptType = "FILE";
                // encrypt_meta.EncryptedTopics
                encrypt_meta.StepSize = temp_raw_key.size() * 8;
                encrypt_meta.KeySize = temp_raw_key.size() * 8;
            }
        } else {
            ifs.open(upload_file, std::ifstream::binary);
            if (!ifs.is_open()) {
                DC_SERVER_LOG_ERROR << "AdvcUploadTask: " << upload_file << " open error";
                return TaskResult::FILE_IO_ERROR;
            }
            ifs.seekg(0, std::ios::end);
            file_size = (uint64_t)ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            DC_SERVER_LOG_DEBUG << "AdvcUploadTask: uploadSingleFile size " << file_size;
            if (file_size == 0) {
                DC_SERVER_LOG_ERROR << "AdvcUploadTask: " << upload_file << " is empty";
                return TaskResult::EMPTY_FILE;
            }
        }
        uint64_t time_sec = getTimeSpecificSec();
        // Calculate max send time. Assume at least 100KBps.
        uint32_t send_seconds_max = file_size / 1024 / 100;
        if (send_seconds_max < 40) {
            send_seconds_max = 40;
        }
        DC_SERVER_LOG_DEBUG << "AdvcUploadTask: uploadSingleFile send_seconds_max " << send_seconds_max;
        static bool setup_res = [send_seconds_max]() -> bool {
            advc::AdvcSysConfig::SetConnTimeoutInms(5 * 1000);
            advc::AdvcSysConfig::SetRecvTimeoutInms(40 * 1000);
            advc::AdvcSysConfig::SetSendTimeoutInms(40 * 1000);
            advc::AdvcSysConfig::SetUploadConnTimeoutInms(5 * 1000);
            advc::AdvcSysConfig::SetUploadRecvTimeoutInms(40 * 1000);
            advc::AdvcSysConfig::SetUploadSendTimeoutInms(send_seconds_max * 1000);
            advc::AdvcSysConfig::SetUploadPartConnTimeoutInms(5 * 1000);
            advc::AdvcSysConfig::SetUploadPartRecvTimeoutInms(40 * 1000);
            uint32_t part_send_seconds_max = 5 * 1024 / 100;  // Part max size is 5MB.
            DC_SERVER_LOG_DEBUG << "AdvcUploadTask: uploadSingleFile part_send_seconds_max " << part_send_seconds_max;
            advc::AdvcSysConfig::SetUploadPartSendTimeoutInms(part_send_seconds_max * 1000);
            return true;
        }();
        const std::string file_digest = "test";
        const int expire_sec = 3600;
        const std::string channel = "VEHICLE_UPLOAD";
        DC_SERVER_LOG_INFO << "AdvcUploadTask: uploadSingleFile " << ", file_size " << file_size << ", file_path "
                           << upload_file << ", file_digest " << file_digest << ", time_sec " << time_sec
                           << ", upload_type " << *m_upload_type << ", channel " << channel << ", expire_sec "
                           << expire_sec << ", protocol_id " << *m_protocol_id << ", enable_encrypt "
                           << m_enable_encrypt << ", encrypt_meta.AlgorithmMode " << encrypt_meta.AlgorithmMode
                           << ", encrypt_meta.KeyVersionId " << encrypt_meta.KeyVersionId << ", encrypt_meta.StepSize "
                           << encrypt_meta.StepSize << ", encrypt_meta.KeySize " << encrypt_meta.KeySize;
        std::istream* ist{nullptr};
        if (m_enable_encrypt) {
            ist = &iss;
        } else {
            ist = &ifs;
        }
        advc::UploadFileReq req(upload_token, ist, file_size, upload_file, file_digest, time_sec,
                                m_upload_type->c_str(), channel, expire_sec, *m_protocol_id, m_enable_encrypt,
                                encrypt_meta);
        advc::UploadFileResp resp = !m_checkpoint->empty()
                                        ? advc_client.ResumableUploadFile(req, *m_checkpoint, *stop_flag)
                                        : advc_client.UploadFile(req, *stop_flag);
        DC_SERVER_LOG_INFO << "AdvcUploadTask: upload file result: " << resp.is_success
                           << ", file_name: " << upload_file << ", checkpoint dir: " << *m_checkpoint;
        if (!resp.is_success) {
            DC_SERVER_LOG_ERROR << "AdvcUploadTask: upload file result: " << resp.is_success
                                << ", error.codeN: " << resp.error.codeN << ", error: " << resp.error.ToJsonString();
            if (strcmp(resp.error.code.c_str(), "UploadCertificateExpireError") == 0 && !m_checkpoint->empty()) {
                // Delete advc checkpoint files due to upload certificate expiration.
                std::filesystem::path dir{*m_checkpoint};
                auto del_count = std::filesystem::remove_all(dir);
                DC_SERVER_LOG_DEBUG << "AdvcUploadTask: remove checkpoint dir " << *m_checkpoint;
            }
        }
        // check whether upload token is valid.
        if (resp.error.codeN == ADVC_RESP_CODE_TOKEN_EXPIRED) {
            DC_SERVER_LOG_ERROR << "AdvcUploadTask: upload token expired";
            upload_token = get_upload_token();
        } else if (upload_token.empty()) {
            DC_SERVER_LOG_ERROR << "AdvcUploadTask: upload token empty";
            upload_token = get_upload_token();
        }

        if (resp.is_success) {
            return TaskResult::OK;
        }
        if (get_stop_flag()) {
            return TaskResult::UPLOAD_FAILED;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds((i + 1) * 2 * m_retry_interval));
    }
    return TaskResult::UPLOAD_FAILED;
}

}  // namespace dc
}  // namespace netaos
}  // namespace hozon
