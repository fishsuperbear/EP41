/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: phm fault dispatcher
 */
#include <string>
#include "proxy.h"
#include "skeleton.h"
#include "phmPubSubTypes.h"

#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/fault_manager/analysis/fault_analysis.h"
#include "phm_server/include/fault_manager/file/phm_file_operate.h"
#include "phm_server/include/fault_manager/manager/phm_collect_file.h"

namespace hozon {
namespace netaos {
namespace phm_server {

const std::string REQ_EVENT_TOPIC = "ReqCollectFileTopic";
const std::string RESP_EVENT_TOPIC = "RespCollectFileTopic";
std::vector<FileName> tempAllFiles;


PhmCollectFile::PhmCollectFile()
{
    PHMS_INFO << "PhmCollectFile::PhmCollectFile";
}

PhmCollectFile::~PhmCollectFile()
{
    PHMS_INFO << "PhmCollectFile::~PhmCollectFile";
}

void
PhmCollectFile::Init()
{
    PHMS_INFO << "PhmCollectFile::Init";
    // init recv
    recv_pubsubtype_ = std::make_shared<RequstTypePubSubType>();
    proxy_ = std::make_shared<Proxy>(recv_pubsubtype_);
    proxy_->Init(0, REQ_EVENT_TOPIC);
    proxy_->Listen(std::bind(&PhmCollectFile::ReceiveReq, this));

    // init send
    send_pubsubtype_ = std::make_shared<CollectionFilePubSubType>();
    skeleton_ = std::make_shared<Skeleton>(send_pubsubtype_);
    skeleton_->Init(0, RESP_EVENT_TOPIC);
}

void
PhmCollectFile::DeInit()
{
    PHMS_INFO << "PhmCollectFile::DeInit";
    if (skeleton_ != nullptr) {
        skeleton_->Deinit();
        skeleton_ = nullptr;
    }

    if (proxy_ != nullptr) {
        proxy_->Deinit();
        proxy_ = nullptr;
    }
}

void
PhmCollectFile::ReceiveReq()
{
    PHMS_DEBUG << "PhmCollectFile::ReceiveReq.";
    if (nullptr == proxy_) {
        PHMS_ERROR << "PhmCollectFile::ReceiveReq proxy_ is nullptr.";
        return;
    }

    if (!proxy_->IsMatched()) {
        PHMS_WARN << "PhmCollectFile::ReceiveReq proxy_ not matched.";
        return;
    }

    std::shared_ptr<RequstType> data = std::make_shared<RequstType>();
    proxy_->Take(data);
    PHMS_INFO << "PhmCollectFile::ReceiveReq type: " << data->type();

    // thread get and send
    std::thread getFileThd([this]{
        FaultAnalysis::getInstance()->UpdateAnalyFile();

        std::vector<std::string> otuAllFiles;
        FileOperate::getInstance()->GetCollectData(otuAllFiles);
        for (auto& file : otuAllFiles) {
            PHMS_INFO << "PhmCollectFile::ReceiveReq file:" << file;
        }

        tempAllFiles.clear();
        for (auto& f : otuAllFiles) {
            FileName cFileName;
            cFileName.fileName(f);
            tempAllFiles.push_back(cFileName);
        }

        ResposeCollectFile(tempAllFiles);
    });
    getFileThd.detach();
}

void
PhmCollectFile::ResposeCollectFile(std::vector<FileName>& tempAllFiles)
{
    PHMS_DEBUG << "PhmCollectFile::ResposeCollectFile.";
    if (nullptr == skeleton_) {
        PHMS_ERROR << "PhmCollectFile::ResposeCollectFile skeleton_ is nullptr.";
        return;
    }

    if (!skeleton_->IsMatched()) {
        PHMS_WARN << "PhmCollectFile::ResposeCollectFile skeleton_ not matched.";
        return;
    }

    std::shared_ptr<CollectionFile> data = std::make_shared<CollectionFile>();
    data->allFileNames(tempAllFiles);
    if (skeleton_->Write(data) == 0) {
        PHMS_INFO << "PhmCollectFile::ResposeCollectFile send ok";
    }
}


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
