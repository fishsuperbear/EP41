#pragma once

#include <iostream>
#include <memory>
#include <unistd.h>
#include "cm/include/skeleton.h"
#include "idl/generated/avmPubSubTypes.h"
#include "log/include/logging.h"

using namespace hozon::netaos::cm;

class app_pub
{
public:
    app_pub(){};
    ~app_pub(){};
    void init(const std::string& topic);
    void run(int interval);
    void set_interval(int interval);
    void deinit(void);
private:
    std::shared_ptr<avmPubSubType> pubsubtype_;
    std::shared_ptr<Skeleton> skeleton_;

    bool _stop_flag = false;
    int interval_ = 0;
    // std::shared_ptr<hozon::netaos::log::Logger> publog{hozon::netaos::log::CreateLogger("app_pub", "orin app pub",
    //                                                 hozon::netaos::log::LogLevel::kInfo)};
};

void app_pub::init(const std::string& topic)
{
    pubsubtype_ = std::make_shared<avmPubSubType>();
    skeleton_ = std::make_shared<Skeleton>(pubsubtype_);
    skeleton_->Init(0, topic);
}

void app_pub::deinit(void)
{
    _stop_flag = true;
    skeleton_->Deinit();
}

void app_pub::run(int interval)
{
    interval_ = interval;
    std::shared_ptr<avm> data = std::make_shared<avm>();
    data->name("avmTopic");
    int count = 0;
    while(!_stop_flag) {
        data->header().seq(count++);
        if (skeleton_->IsMatched()) {
            if (skeleton_->Write(data) == 0) {
                // publog->LogInfo() << " Send data.name : " << data->name() << " data.seq : " << data->header().seq();
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_));
    }
}


void app_pub::set_interval(int interval)
{
    interval_ = interval;
}



