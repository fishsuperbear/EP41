/*
* Copyright (c) Hozon Auto Co., Ltd. 2023-2023. All rights reserved.
* Description: uds raw data event receiver
*/


#include "uds_current_session_receiver.h"
#include "update_manager/update_check/update_check.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {


UdCurrentSessionReceiver::UdCurrentSessionReceiver()
: proxy_(nullptr)
, data_(nullptr)
{
}

UdCurrentSessionReceiver::~UdCurrentSessionReceiver()
{
}

void
UdCurrentSessionReceiver::Init()
{
    UPDATE_LOG_I("UdCurrentSessionReceiver::Init");
    std::shared_ptr<uds_current_session_notify_eventPubSubType> pubsubtype_ = std::make_shared<uds_current_session_notify_eventPubSubType>();
    proxy_ = std::make_shared<Proxy>(pubsubtype_);
    proxy_->Init(0, "uds_current_session_notify_eventTopic");
    data_ = std::make_shared<uds_current_session_notify_event>();

    proxy_->Listen(std::bind(&UdCurrentSessionReceiver::EventCallback, this));
}

void
UdCurrentSessionReceiver::DeInit()
{
    UPDATE_LOG_I("UdCurrentSessionReceiver::DeInit");
    if (nullptr != proxy_) {
        proxy_->Deinit();
        proxy_ = nullptr;
    }
    data_ = nullptr;
}

void
UdCurrentSessionReceiver::EventCallback()
{
    if (proxy_ != nullptr && proxy_->IsMatched()) {
        proxy_->Take(data_);
        
        UPDATE_LOG_D("UdCurrentSessionReceiver, data: [%d].", static_cast<uint16_t>(data_->current_session()));
        if (data_->current_session() == 0x02) {
            UPDATE_LOG_D("UpdateManager switch into Update Mode.");
            UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_OTA);
        } else if (data_->current_session() == 0x01) {
            UPDATE_LOG_D("UpdateManager switch into Normal Mode.");
            UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_NORMAL);
        } else {
            // go on
        }
    }
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
