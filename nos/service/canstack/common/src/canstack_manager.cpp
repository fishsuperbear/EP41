/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: socket can interface canstack manager
*/


#include <mutex>

#include "canstack_manager.h"
#include "config_loader.h"
// #include "can_stack_utils.h"
#include "canstack_logger.h"


namespace hozon {
namespace netaos {
namespace canstack {

CanStackManager* CanStackManager::sinstance_ = nullptr;
std::mutex g_mutex;

CanStackManager *CanStackManager::Instance()
{
    if (nullptr == sinstance_)
    {
        std::lock_guard<std::mutex> lck(g_mutex);
        if (nullptr == sinstance_)
        {
            sinstance_ = new CanStackManager();
        }
    }
    return sinstance_;
}

CanStackManager::CanStackManager() : publisher_(nullptr), subscriber_(nullptr)
{
    can_monitor_ = std::make_shared<hozon::netaos::canstack::CanbusMonitor>();
}

CanStackManager::~CanStackManager()
{
}

int CanStackManager::Init(const std::string &canDevice, CanParser* canParser, 
    Publisher* publisher, Subscriber* subscriber)
{
    CAN_LOG_INFO << "CanStackManager::Init enter...";
    int res = can_monitor_->Init(canDevice, canParser);
    if (res > 0) {
        if (publisher) {
            publisher_ = publisher;
            publisher_->Init();
        }

        if (subscriber) {
            subscriber_ = subscriber;
            subscriber_->Init();
        }
    }

    return res;
}

int CanStackManager::Init(const std::string &canDevice, CanParser* canParser, 
        Publisher* publisher, std::vector<Subscriber*> subscriber_list)
{
    CAN_LOG_INFO << "CanStackManager::Init enter...";
    int res = can_monitor_->Init(canDevice, canParser);
    if (res > 0) {
        if (publisher) {
            publisher_ = publisher;
            publisher_->Init();
        }

        subscriber_list_ = subscriber_list;
        for (auto subscriber : subscriber_list) {
            if (subscriber) {
                subscriber->Init();
            }
        }
    }

    return res;
}

int CanStackManager::Init(const std::string &canDevice, CanParser* canParser, 
        std::vector<Publisher*> publisher_list, Subscriber* subscriber)
{
    CAN_LOG_INFO << "CanStackManager::Init enter...";
    int res = can_monitor_->Init(canDevice, canParser);
    if (res > 0) {
        publisher_list_ = publisher_list;
        for (auto publisher : publisher_list) {
            if (publisher) {
                publisher->Init();
            }
        }

        if (subscriber) {
            subscriber_ = subscriber;
            subscriber_->Init();
        }
    }

    return res;
}
int CanStackManager::Init(const std::vector<std::string> &canDevice, CanParser* canParser, 
        Publisher* publisher, Subscriber* subscriber)
{
    CAN_LOG_INFO << "CanStackManager::Init enter...";
    int res = can_monitor_->Init(canDevice, canParser);
    if (res > 0) {
        if (publisher) {
            publisher_ = publisher;
            publisher_->Init();
        }

        if (subscriber) {
            subscriber_ = subscriber;
            subscriber_->Init();
        }
    }

    return res;
}
int CanStackManager::Init(const std::string &canDevice, CanParser* canParser, 
        std::vector<Publisher*> publisher_list, std::vector<Subscriber*> subscriber_list)
{
    CAN_LOG_INFO << "CanStackManager::Init enter...";
    int res = can_monitor_->Init(canDevice, canParser);
    if (res > 0) {
        publisher_list_ = publisher_list;
        for (auto publisher : publisher_list) {
            if (publisher) {
                publisher->Init();
            }
        }

        subscriber_list_ = subscriber_list;
        for (auto subscriber : subscriber_list) {
            if (subscriber) {
                subscriber->Init();
            }
        }
    }

    return res;
}

void CanStackManager::Start()
{
    CAN_LOG_INFO << "CanStackManager::Start enter...";
    if (hozon::netaos::canstack::ConfigLoader::analysys_on_) {
        // hozon::canstack::CanStackUtils::CreatePerfLogger("PERF", "performance analysis", ara::log::LogLevel::kVerbose);
    }

    if (can_monitor_) {
        can_monitor_->StartCanbusMonitorThread();
    }
    if (publisher_) {
        publisher_->Pub();
    }
    if (subscriber_) {
        subscriber_->Sub();
    }
    for (auto publisher : publisher_list_) {
        if (publisher) {
            publisher->Pub();
        }
    }
    for (auto subscriber : subscriber_list_) {
        if (subscriber) {
            subscriber->Sub();
        }
    }
}

void CanStackManager::Stop()
{
    CAN_LOG_INFO << "CanStackManager::Stop enter...";

    can_monitor_ = nullptr;
    if (publisher_) {
        publisher_->Stop();
        publisher_ = nullptr;
    }
    if (subscriber_) {
        subscriber_->Stop();
        subscriber_ = nullptr;
    }
    for (auto publisher : publisher_list_) {
        if (publisher) {
            publisher->Stop();
        }
    }
    for (auto subscriber : subscriber_list_) {
        if (subscriber) {
            subscriber->Stop();
        }
    }
}

void CanStackManager::SetFilters(const int can_fd, const std::vector<can_filter> &filters)
{
    if (can_monitor_) {
        can_monitor_->SetCanFilters(can_fd, filters);
    }
}

std::string CanStackManager::GetCurrCanDevice(int fd)
{
    std::string canDevice;
    if (can_monitor_) {
        canDevice = can_monitor_->GetCurrCanDevice(fd);
    }
    return canDevice;
}


}  // namespace canstack
}
}  // namespace hozon
