/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: fm debounce base
 */

#pragma once
#include "proxy.h"
#include "skeleton.h"
#include "phmPubSubTypes.h"
#include <stdint.h>
#include <memory>


namespace hozon {
namespace netaos {
namespace phm {
using namespace hozon::netaos::cm;

enum RequestType
{
    RequestTypeGetCollectFile = 0,
};


class FaultCollectionFile
{
public:
    FaultCollectionFile();
    ~FaultCollectionFile() {}

    void Init();
    void Deinit();
    int32_t Request(const int inReqTypeData, std::function<void(std::vector<std::string>&)> collectionFileCb);
    void Recv();

    std::shared_ptr<RequstTypePubSubType> pubsubtype_ {nullptr};
    std::shared_ptr<Skeleton> skeleton_ {nullptr};

    std::shared_ptr<CollectionFilePubSubType> collection_file_pubsubtype_ {nullptr};
    std::shared_ptr<Proxy> proxy_ {nullptr};
    std::shared_ptr<CollectionFile> resp_data_ {nullptr};

    std::function<void(std::vector<std::string>&)> collectionFileCb_{nullptr};
};


}  // namespace phm
}  // namespace netaos
}  // namespace hozon

