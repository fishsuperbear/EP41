
#include <iostream>
#include "phm/common/include/phm_logger.h"
#include "phm/fault_manager/include/fault_collection_file.h"

namespace hozon {
namespace netaos {
namespace phm {

const std::string REQ_EVENT_TOPIC = "ReqCollectFileTopic";
const std::string RESP_EVENT_TOPIC = "RespCollectFileTopic";


FaultCollectionFile::FaultCollectionFile()
{
}

void
FaultCollectionFile::Init()
{
    PHM_INFO << "FaultCollectionFile::Init";
    pubsubtype_ = std::make_shared<RequstTypePubSubType>();
    skeleton_ = std::make_shared<Skeleton>(pubsubtype_);
    skeleton_->Init(0, REQ_EVENT_TOPIC);

    resp_data_ = std::make_shared<CollectionFile>();
    collection_file_pubsubtype_ = std::make_shared<CollectionFilePubSubType>();
    proxy_ = std::make_shared<Proxy>(collection_file_pubsubtype_);
    int32_t res = proxy_->Init(0, RESP_EVENT_TOPIC);
    if (0 == res) {
        proxy_->Listen(std::bind(&FaultCollectionFile::Recv, this));
    }
}

int32_t
FaultCollectionFile::Request(const int inReqTypeData, std::function<void(std::vector<std::string>&)> collectionFileCb)
{
    collectionFileCb_ = collectionFileCb;

    std::shared_ptr<RequstType> req_data = std::make_shared<RequstType>();
    req_data->type(inReqTypeData);

    if (skeleton_ != nullptr && skeleton_->IsMatched()) {
        if (skeleton_->Write(req_data) == 0) {
            PHM_INFO << "FaultCollectionFile::Send type:" << inReqTypeData;
        }
    }
    else {
        PHM_INFO << "FaultCollectionFile::Send skeleton is null or skeleton not matched!";
    }

    return 0;
}

void
FaultCollectionFile::Recv()
{
    if (proxy_ != nullptr && proxy_->IsMatched()) {
        proxy_->Take(resp_data_);
    }

    std::vector<std::string> outAllFileData;
    std::vector<FileName>& cAllFileNames = resp_data_->allFileNames();
    for (auto& fn : cAllFileNames) {
        PHM_INFO << "FaultCollectionFile::Recv fileName:" << fn.fileName();
        outAllFileData.emplace_back(fn.fileName());
    }

    if (collectionFileCb_ != nullptr) {
        collectionFileCb_(outAllFileData);
    }
}

void
FaultCollectionFile::Deinit()
{
    if (skeleton_ != nullptr) {
        skeleton_->Deinit();
        skeleton_ = nullptr;
    }

    if (proxy_ != nullptr) {
        proxy_->Deinit();
        proxy_ = nullptr;
    }
}


}  // namespace phm
}  // namespace netaos
}  // namespace hozon
