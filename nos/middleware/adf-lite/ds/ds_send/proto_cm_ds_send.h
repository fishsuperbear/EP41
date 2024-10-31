#pragma once


#include "adf-lite/ds/ds_send/ds_send.h"
#include "adf-lite/include/executor.h"
#include "cm/include/skeleton.h"

namespace hozon {
namespace netaos {
namespace adf_lite {
    
class ProtoCMDsSend : public DsSend {
public:
    ProtoCMDsSend(const DSConfig::DataSource& config);
    virtual ~ProtoCMDsSend();
    virtual void PreDeinit() override;
    virtual void Deinit() override;
    virtual void PauseSend() override;
    virtual void ResumeSend() override;

private:
    bool _stop{false};
    hozon::netaos::cm::Skeleton _skeleton;
    std::shared_ptr<std::thread> _recv;
    int32_t Write(std::shared_ptr<google::protobuf::Message> data, Header& header);
    void ReceiveInnerTopicSendCmTopic();
};

}
}
}