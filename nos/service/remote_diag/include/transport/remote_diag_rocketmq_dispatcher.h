#ifndef REMOTE_DIAG_ROCKETMQ_DISPATCHER_H
#define REMOTE_DIAG_ROCKETMQ_DISPATCHER_H

#include <thread>
#include "json/json.h"

#include "DefaultMQProducer.h"
#include "DefaultMQPushConsumer.h"
#include "remote_diag/include/common/remote_diag_def.h"

namespace hozon {
namespace netaos {
namespace remote_diag {

using namespace std;
using namespace rocketmq;

class RocketMQMsgListener : public MessageListenerConcurrently {
public:
    RocketMQMsgListener() {}
    virtual ~RocketMQMsgListener() {}

    virtual ConsumeStatus consumeMessage(const std::vector<MQMessageExt>& msgs);
};

class RemoteDiagRocketMQDispatcher {
public:
    RemoteDiagRocketMQDispatcher();
    ~RemoteDiagRocketMQDispatcher();

    void Init();
    void DeInit();

    void SendMessage(const Json::Value& message);

private:
    DefaultMQProducer* mq_producer_;
    DefaultMQPushConsumer* mq_consumer_;

};

}  // namespace remote_diag
}  // namespace netaos
}  // namespace hozon
#endif  // REMOTE_DIAG_ROCKETMQ_DISPATCHER_H
