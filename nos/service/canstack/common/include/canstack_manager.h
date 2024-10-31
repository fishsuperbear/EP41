/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface canstack manager
 */

#ifndef CANSTACK_MANAGER_H
#define CANSTACK_MANAGER_H

#include <string>
#include <vector>

#include "can_parser.h"
#include "canbus_monitor.h"
#include "publisher.h"
#include "subscriber.h"

namespace hozon {
namespace netaos {
namespace canstack {

class CanStackManager {
public:
    static CanStackManager* Instance();
    ~CanStackManager();

    int Init(const std::string& canDevice, CanParser* canParser,
           Publisher* publisher, Subscriber* subscriber);

    int Init(const std::string& canDevice, CanParser* canParser,
           Publisher* publisher, std::vector<Subscriber*> subscriber_list);

    int Init(const std::string& canDevice, CanParser* canParser,
           std::vector<Publisher*> publisher_list, Subscriber* subscriber);

    int Init(const std::vector<std::string> &canDevice, CanParser* canParser, 
        Publisher* publisher, Subscriber* subscriber);

    int Init(const std::string& canDevice, CanParser* canParser,
           std::vector<Publisher*> publisher_list,
           std::vector<Subscriber*> subscriber_list);

    void Start();

    void Stop();

    void SetFilters(const int can_fd, const std::vector<can_filter>& filters);

    std::string GetCurrCanDevice(int fd);

private:
    CanStackManager();

    static CanStackManager* sinstance_;
    std::shared_ptr<hozon::netaos::canstack::CanbusMonitor> can_monitor_;
    Publisher* publisher_;
    Subscriber* subscriber_;
    std::vector<Publisher*> publisher_list_;
    std::vector<Subscriber*> subscriber_list_;

};

}  // namespace canstack
}
}  // namespace hozon
#endif  // CANSTACK_MANAGER_H
