#pragma once

#include "cm/include/skeleton.h"
#include "idl/generated/diagPubSubTypes.h"
#include "log/include/default_logger.h"

#include <unistd.h>
#include <iostream>
#include <memory>
#include <thread>

using namespace hozon::netaos::cm;
const uint32_t DOMAIN = 0;
const std::string TOPIC = "reportDemEvent";

class DemEventPub
{
public:
    DemEventPub() { init(); };
    ~DemEventPub() { deinit(); };

    void init();
    void runOccur();
    void runRecover();
    void deinit();

private:
    std::shared_ptr<reportDemEventPubSubType> pubsubtype_;
    std::shared_ptr<Skeleton> skeleton_;

    bool _stop_flag = false;
};

void DemEventPub::init()
{
    printf("DemEventPub::init\n");
    pubsubtype_ = std::make_shared<reportDemEventPubSubType>();
    skeleton_ = std::make_shared<Skeleton>(pubsubtype_);
    skeleton_->Init(DOMAIN, TOPIC);
}

void DemEventPub::deinit()
{
    printf("DemEventPub::deinit\n");
    _stop_flag = true;
    skeleton_->Deinit();
}

void DemEventPub::runOccur()
{
    printf("runOccur start\n");
    std::shared_ptr<reportDemEvent> data = std::make_shared<reportDemEvent>();
    uint32_t dtcValue = 0x123456;
    uint8_t alarmStatus = 1;
    data->dtcValue(dtcValue);
    data->alarmStatus(alarmStatus);

    bool isSend = false;
    while (1) {
        if (!skeleton_->IsMatched()) {
            printf("runOccur skeleton_ IsMatched false\n");
            sleep(1);
            continue;
        }

        if (isSend) {
            sleep(1);
            continue;
        }

        printf("runOccur skeleton_ IsMatched true\n");
        if (skeleton_->Write(data) == 0) {
            printf("runOccur data.dtcValue:0x%x, alarmStatus:0x%02x\n", data->dtcValue(), data->alarmStatus());
            break;
        }

        sleep(1);
    }
    return;
}

void DemEventPub::runRecover()
{
    printf("runRecover start\n");
    std::shared_ptr<reportDemEvent> data = std::make_shared<reportDemEvent>();
    uint32_t dtcValue = 0x123456;
    uint8_t alarmStatus = 0;
    data->dtcValue(dtcValue);
    data->alarmStatus(alarmStatus);

    while (1) {
        if (!skeleton_->IsMatched()) {
            printf("runRecover skeleton_ IsMatched false\n");
            sleep(1);
            continue;
        }

        printf("runRecover skeleton_ IsMatched true\n");
        if (skeleton_->Write(data) == 0) {
            printf("runRecover data.dtcValue:0x%x, alarmStatus:0x%02x\n", data->dtcValue(), data->alarmStatus());
            break;
        }

        sleep(1);
    }

    return;
}