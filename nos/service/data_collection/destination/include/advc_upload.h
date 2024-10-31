/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: advc_upload.h
 * @Date: 2023/08/17
 * @Author: cheng
 * @Desc: --
 */

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "advc_client.h"
#include "basic/basic_task.h"
#include "common/thread_pool/include/thread_pool_flex.h"

namespace hozon {
namespace netaos {
namespace dc {

struct Configuration_ {
    constexpr static int DEF_RETRY_COUNT = 5;
    constexpr static int DEF_RETRY_INTERVAL = 500;
    // protocolId
    std::string protocolId{""};
    // retryCount
    int retryCount{5};
    // retryInterval
    int retryInterval{500};
    // uploadType
    std::string uploadType{""};
    // checkPoint
    std::string checkPoint{""};
    // enableEncrypt
    bool enableEncrypt{false};
    // deleteAfterUpload
    bool deleteAfterUpload{false};
};

class AdvcUploadTask : public BasicTask {
   public:
    AdvcUploadTask();

    ~AdvcUploadTask();

    void configure(std::string type, YAML::Node& node) override;

    void configure(std::string type, DataTrans& node) override;

    void active() override;

    void deactive() override;

    TaskStatus getStatus() override;

    void setThreadPool(ThreadPoolFlex* threadPool) { m_threadpool = threadPool; }

    bool getTaskResult(const std::string& type, struct DataTrans& dataStruct) override {
        dataStruct.dataType = DataTransType::fileAndFolder;
        return true;
    }

   private:
    enum TaskResult { OK = 0, UPLOAD_FAILED = 1, EMPTY_FILE = 2, FILE_IO_ERROR = 3, REQUEST_ENCRYPTKEY_FAILED = 4, GET_UPLOAD_TOKEN_FAILED = 5 };

    static constexpr uint32_t ADVC_RESP_CODE_TOKEN_EXPIRED = 100026u;
    inline TaskStatus get_status() {
        return m_status->load(std::memory_order_acquire);
    }


    inline void set_status(TaskStatus taskstatus) {
        m_status->store(taskstatus, std::memory_order_release);
    }
    inline bool get_stop_flag() {
        return m_stop_flag->load(std::memory_order_acquire);
    }
    inline void set_stop_flag(bool stop_flag) {
        m_stop_flag->store(stop_flag, std::memory_order_release);
    }
    std::shared_ptr<std::atomic_bool> m_stop_flag;
    std::shared_ptr<std::atomic<TaskStatus>> m_status;
    int error_count{0};
    bool m_enable_encrypt{false};
    bool m_delete_after_upload{false};
    std::shared_ptr<std::vector<std::string>> m_upload_files;
    std::shared_ptr<std::vector<std::string>> m_origin_path;
    std::shared_ptr<std::string> m_toupload_dir;
    std::shared_ptr<std::string> m_protocol_id;
    int m_retry_count{-1};
    int m_retry_interval{-1};
    std::shared_ptr<std::string> m_upload_type;
    std::shared_ptr<std::string> m_checkpoint;
    ThreadPoolFlex* m_threadpool{nullptr};

    std::string get_upload_token();
    bool uploadFiles();
    TaskResult uploadSingleFile(advc::AdvcClient& advc_client, std::string upload_file, std::string upload_token, std::shared_ptr<std::atomic_bool> stop_flag);
    TaskResult requestEncryptKey(std::string& upload_token, std::string& encrpytion_key, std::string& version_id, std::string& raw_key);
    std::string encryptData(std::string encryption_key, std::string raw_key, std::string upload_file);

    static std::string m_encrpytion_key;
    static std::string m_encryption_version_id;
    static std::mutex m_encryption_key_mutex;
    static std::string m_raw_key;
    static std::atomic_bool selfActive_;
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

YCS_ADD_STRUCT(hozon::netaos::dc::Configuration_, protocolId, retryCount, retryInterval, uploadType, checkPoint, enableEncrypt, deleteAfterUpload);
