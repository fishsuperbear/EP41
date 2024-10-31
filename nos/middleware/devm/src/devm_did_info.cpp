/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: devm
 * Description: devm_did_info.cpp
 * Created on: Nov 21, 2023
 * Author: yanlongxiang
 *
 */
#include "devm_logger.h"
#include "devm/include/devm_did_info.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace devm {

// read_dids
DevmClientDidInfo::DevmClientDidInfo() {
}

DevmClientDidInfo::~DevmClientDidInfo() {
}

std::string DevmClientDidInfo::ReadDidInfo(std::string did) {
    DEVM_LOG_INFO << "read-did.";
    client_ = std::make_shared<ZmqIpcClient>();
    client_->Init("tcp://localhost:11122");
    std::string reply{};
    DevmReq req_data{};
    req_data.set_req_type("did_info");
    req_data.set_data_value(did.c_str());
    int32_t res = client_->Request(req_data.SerializeAsString(), reply, 2000);
    if (res < 0) {
        DEVM_LOG_INFO << "DevmClientDidInfo ReadDid Request failed. failedCode: " << res;
        client_->Deinit();
        return "";
    }
    DevmDidInfo resp_data{};
    resp_data.ParseFromString(reply);
    client_->Deinit();
    return resp_data.data_value();
}



}  // namespace devm
}  // namespace netaos
}  // namespace hozon
