/*
* Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
* Description: remote diag rocketmq dispatcher
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <cstddef>
#include <functional>
#include <fstream>
#include <iostream>

#include "remote_diag/include/common/remote_diag_logger.h"
#include "remote_diag/include/common/remote_diag_config.h"
#include "remote_diag/include/transport/remote_diag_rocketmq_dispatcher.h"
#include "remote_diag/include/handler/remote_diag_handler.h"

namespace hozon {
namespace netaos {
namespace remote_diag {


ConsumeStatus
RocketMQMsgListener::consumeMessage(const std::vector<MQMessageExt>& msgs)
{
    for (auto& msgExt : msgs) {
        std::string payload = msgExt.getBody();
        std::string msg = (payload.size() > 100) ? payload.substr(0, 100) + ("...") : payload;
        DGR_INFO << "RocketMQMsgListener::consumeMessage topic: " << msgExt.getTopic() << " message: " << msg;
        Json::Value reqMessage;
        Json::String errs;
        Json::CharReaderBuilder reader;
        std::unique_ptr<Json::CharReader> const jsonReader(reader.newCharReader());
        if (!jsonReader->parse(payload.c_str(), payload.c_str() + payload.length(), &reqMessage, &errs))
        {
            DGR_ERROR << "RemoteDiagRocketMQDispatcher::consumeMessage error data format.";
            continue;
        }

        auto itr = find(REMOTE_DIAG_REQUEST_TYPE.begin(), REMOTE_DIAG_REQUEST_TYPE.end(), reqMessage["TYPE"].asString());
        if ((reqMessage["DATA"].asString() == "") || (itr == REMOTE_DIAG_REQUEST_TYPE.end())) {
            continue;
        }

        RemoteDiagHandler::getInstance()->RecvRemoteMessage(reqMessage);
    }

    return CONSUME_SUCCESS;
}

RemoteDiagRocketMQDispatcher::RemoteDiagRocketMQDispatcher()
    : mq_producer_(new DefaultMQProducer("RemoteDiagReq"))
    , mq_consumer_(new DefaultMQPushConsumer("RemoteDiagRes"))
{
}

RemoteDiagRocketMQDispatcher::~RemoteDiagRocketMQDispatcher()
{
    if (nullptr != mq_producer_) {
        delete mq_producer_;
        mq_producer_ = nullptr;
    }

    if (nullptr != mq_consumer_) {
        delete mq_consumer_;
        mq_consumer_ = nullptr;
    }
}

void
RemoteDiagRocketMQDispatcher::Init()
{
    DGR_INFO << "RemoteDiagRocketMQDispatcher::Init";
    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    // consumer init
    mq_consumer_->setNamesrvAddr(configInfo.RocketMQAddress);
    mq_consumer_->setGroupName(configInfo.RocketMQReqGroup);
    mq_consumer_->setSessionCredentials(configInfo.RocketMQAccessKey, configInfo.RocketMQSecretKey, "");
    mq_consumer_->setNamesrvDomain(configInfo.RocketMQDomain);

    mq_consumer_->setInstanceName(configInfo.RocketMQReqGroup);
    mq_consumer_->subscribe(configInfo.RocketMQReqTopic, "*");
    mq_consumer_->setConsumeThreadCount(15);
    mq_consumer_->setTcpTransportTryLockTimeout(1000);
    mq_consumer_->setTcpTransportConnectTimeout(400);

    RocketMQMsgListener msglistener;
    mq_consumer_->registerMessageListener(&msglistener);

    try {
        mq_consumer_->start();
    } catch (MQException& e) {
        DGR_ERROR << "RemoteDiagRocketMQDispatcher::Init mq_consumer start failed ErrorCode: " << e.GetError() << ", Exception:" << e.what();
    }

    // producer init
    mq_producer_->setSessionCredentials(configInfo.RocketMQAccessKey, configInfo.RocketMQSecretKey, "");
    mq_producer_->setTcpTransportTryLockTimeout(1000);
    mq_producer_->setTcpTransportConnectTimeout(400);
    mq_producer_->setNamesrvDomain(configInfo.RocketMQDomain);
    mq_producer_->setNamesrvAddr(configInfo.RocketMQAddress);
    mq_producer_->setGroupName(configInfo.RocketMQResGroup);

    try {
        mq_producer_->start();
    } catch (MQException& e) {
        DGR_ERROR << "RemoteDiagRocketMQDispatcher::Init mq_producer start failed ErrorCode: " << e.GetError() << ", Exception:" << e.what();
    }
}

void
RemoteDiagRocketMQDispatcher::DeInit()
{
    DGR_INFO << "RemoteDiagRocketMQDispatcher::DeInit";
    if (nullptr != mq_producer_) {
        mq_producer_->shutdown();
    }

    if (nullptr != mq_consumer_) {
        mq_consumer_->shutdown();
    }

    DGR_INFO << "RemoteDiagRocketMQDispatcher::DeInit finish!";
}

void
RemoteDiagRocketMQDispatcher::SendMessage(const Json::Value& message)
{
    DGR_INFO << "RemoteDiagRocketMQDispatcher::SendMessage.";
    if (nullptr == mq_producer_) {
        return;
    }

    std::string payload;
    Json::StreamWriterBuilder wbuilder;
    wbuilder["indentation"] = "";
    payload = Json::writeString(wbuilder, message);

    const RemoteDiagConfigInfo& configInfo = RemoteDiagConfig::getInstance()->GetRemoteDiagConfigInfo();
    MQMessage msg(
            configInfo.RocketMQResTopic,
            configInfo.RocketMQResTag,
            configInfo.RocketMQResKeys,
            payload
    );

    try {
        SendResult sendResult = mq_producer_->send(msg);
        std::string msg = (payload.size() > 100) ? payload.substr(0, 100) + ("...") : payload;
        DGR_INFO << "RemoteDiagRocketMQDispatcher::SendMessage result: " << sendResult.getSendStatus() << ", ID: " << sendResult.getMsgId() << ", msg: " << msg;
    } catch (MQException& e) {
        DGR_ERROR << "RemoteDiagRocketMQDispatcher::SendMessage failed ErrorCode: " << e.GetError() << ", Exception:" << e.what();
    }
}

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon