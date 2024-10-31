/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_client.cpp
 * @Date: 2023/10/27
 * @Author: cheng
 * @Desc: --
 */

#include "client/include/dc_client.h"

#include <cstdint>
#include <string>
#include <vector>

#include "client/include/dc_client_impl.h"

namespace hozon {
namespace netaos {
namespace dc {

DcClient::DcClient(){
    DcClientImpl *p = new DcClientImpl();
    m_client = reinterpret_cast<void *>(p);
};

DcClient::~DcClient(){
    DcClientImpl *p = reinterpret_cast<DcClientImpl *>(m_client) ;
    delete p;
    m_client = nullptr;
};

DcResultCode DcClient::Init(const std::string client_name, const uint32_t max_wait_millis){
    return reinterpret_cast<DcClientImpl*>(m_client)->Init(client_name, max_wait_millis);
};

DcResultCode DcClient::DeInit(){
    return reinterpret_cast<DcClientImpl*>(m_client)->DeInit();
};

DcResultCode DcClient::CollectTrigger(uint32_t trigger_id) {
    return reinterpret_cast<DcClientImpl*>(m_client)->CollectTrigger(trigger_id);
};

DcResultCode DcClient::CollectTriggerDesc(uint32_t trigger_id, uint64_t time){
    return reinterpret_cast<DcClientImpl*>(m_client)->CollectTriggerDesc(trigger_id, time);
}

DcResultCode DcClient::CollectTriggerDesc(uint32_t trigger_id, uint64_t time, std::string desc){
    return reinterpret_cast<DcClientImpl*>(m_client)->CollectTriggerDesc(trigger_id, time, desc);
}

DcResultCode DcClient::Upload(std::vector<std::string> &path_list, std::string file_type, std::string file_name, uint16_t cache_file_num){
    return reinterpret_cast<DcClientImpl*>(m_client)->Upload(path_list, file_type, file_name,cache_file_num);
};

DcResultCode DcClient::Upload(std::vector<char> &data, std::string file_type, std::string file_name, uint16_t cache_file_num){
    return reinterpret_cast<DcClientImpl*>(m_client)->Upload(data, file_type, file_name,cache_file_num);
};


}  // namespace dc
}  // namespace netaos
}  // namespace hozon
