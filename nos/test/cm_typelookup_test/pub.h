#pragma once

#include <iostream>
#include <memory>
#include <unistd.h>
#include "cm/include/skeleton.h"
#include "idl/generated/avmPubSubTypes.h"
#include "log/include/default_logger.h"

using namespace hozon::netaos::cm;

class pub
{
public:
    pub(){};
    ~pub(){};
    void init(const std::string& topic);
    void run();
    void deinit(void);
private:
    std::shared_ptr<avmPubSubType> pubsubtype_;
    std::shared_ptr<Skeleton> skeleton_;

    bool _stop_flag = false;
};

void pub::init(const std::string& topic)
{
    pubsubtype_ = std::make_shared<avmPubSubType>();
    skeleton_ = std::make_shared<Skeleton>(pubsubtype_);
    skeleton_->Init(0, topic);
}

void pub::deinit(void)
{
    _stop_flag = true;
    skeleton_->Deinit();
}

void pub::run()
{
    std::shared_ptr<avm> data = std::make_shared<avm>();
    data->name("avmTopic");
    int count = 0;
    while(!_stop_flag) {
        data->header().seq(count++);
        if (skeleton_->IsMatched()) {
            if (skeleton_->Write(data) == 0) {
                DF_LOG_INFO << " Send data.name : " << data->name() << " data.seq : " << data->header().seq();
            }
        }
        sleep(1);
    }
}



