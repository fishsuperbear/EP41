#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <unistd.h>

#include "cm/include/proxy.h"
#include "idl/generated/avmPubSubTypes.h"
#include "log/include/logging.h"
#include "phm/phm_client.h"

using namespace hozon::netaos::cm;
using namespace hozon::netaos::phm;

using namespace std;
using namespace placeholders;

class app_sub
{
public:
    app_sub();
    ~app_sub();
    void init(const std::string& topic);
    void deinit();
    void run(std::function<void(void)> callback);
    void sub_callback(void);
    void ServiceAvailableCallback(const bool bResult);
    void FaultReceiveCallback(const ReceiveFault_t& fault);

private:
    std::shared_ptr<avmPubSubType> pubsubtype_;
    std::shared_ptr<Proxy> proxy_;

    std::shared_ptr<avm> data;
    bool _stop_flag = false;
    std::shared_ptr<PHMClient> phm_client_;
    std::shared_ptr<hozon::netaos::log::Logger> sublog{hozon::netaos::log::CreateLogger("app_sub", "orin app sub",
                                                    hozon::netaos::log::LogLevel::kInfo)};

};


app_sub::app_sub()
{
}

app_sub::~app_sub()
{
}

void app_sub::init(const std::string& topic) {
    pubsubtype_ = std::make_shared<avmPubSubType>();
    proxy_ = std::make_shared<Proxy>(pubsubtype_);
    proxy_->Init(0, topic);

    data = std::make_shared<avm>();
    phm_client_ = std::make_shared<PHMClient>();
    phm_client_->Init("/storage/rmdiag/plugin/plugin_phm_app_startup/user/conf/app_sub_phm_config.yaml", std::bind(&app_sub::ServiceAvailableCallback, this, _1), std::bind(&app_sub::FaultReceiveCallback, this, _1));
    phm_client_->Start(1000);
}

void app_sub::deinit() {
    _stop_flag = true;
    if (phm_client_ != nullptr) {
        phm_client_->Stop();
        phm_client_->Deinit();
    }
    proxy_->Deinit();
}

void app_sub::run(std::function<void(void)> callback) {
    proxy_->Listen(callback);

    while (!_stop_flag)
    {
        sleep(1);
    }
}

void app_sub::sub_callback() {
    if (proxy_->IsMatched()) {
        if (phm_client_ != nullptr) {
            phm_client_->ReportCheckPoint(0);
        }
        proxy_->Take(data);
        sublog->LogInfo() << " Recv data.name : " << data->name() << " data.seq : " << data->header().seq();
    }
}

void app_sub::ServiceAvailableCallback(const bool bResult) {
    sublog->LogInfo() << "---------------PHM ServiceAvailableCallback bResult: " << bResult << "--------------";
}

void app_sub::FaultReceiveCallback(const ReceiveFault_t& fault) {
    sublog->LogInfo() << "---------------FaultReceiveCallback faultId: " << fault.faultId << " faultObj: " << static_cast<uint>(fault.faultObj)
                                    << " faultStatus: " << fault.faultStatus << " faultOccurTime: " << fault.faultOccurTime
                                    << " faultDomain: " << fault.faultDomain << " faultDes: " << fault.faultDes << "--------------";

    for (auto& item : fault.faultCluster) {
        sublog->LogInfo() << "---------------FaultReceiveCallback clusterName: " << item.clusterName << " clusterValue: " << item.clusterValue;
    }
}
