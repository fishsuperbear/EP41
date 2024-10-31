/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: phm fault dispatcher
 */

#pragma once
#include <string.h>
#include <memory>
#include "proxy.h"
#include "skeleton.h"
#include "phmPubSubTypes.h"


namespace hozon {
namespace netaos {
namespace phm_server {


using namespace hozon::netaos::cm;
class PhmCollectFile
{
public:
    PhmCollectFile();
    ~PhmCollectFile();
    void Init();
    void DeInit();
    void ReceiveReq();
    void ResposeCollectFile(std::vector<FileName>& tempAllFiles);

    // send
    std::shared_ptr<CollectionFilePubSubType> send_pubsubtype_ {nullptr};
    std::shared_ptr<Skeleton> skeleton_ {nullptr};

    // recv
    std::shared_ptr<RequstTypePubSubType> recv_pubsubtype_ {nullptr};
    std::shared_ptr<Proxy> proxy_ {nullptr};
    std::shared_ptr<RequstType> recv_data_ {nullptr};
};


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
