/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: extwdg
*/
#ifndef EXTWDG_MESSAGE_H_
#define EXTWDG_MESSAGE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "extwdg_logger.h"
#include "extwdg_check.h"
#include "extwdg_quest_answer.h"
#include "phm_client_instance.h"
// #include "extwdg_transport.h"

namespace hozon {
namespace netaos {
namespace extwdg {
#pragma pack(1)
struct MessageBuffer
{
    uint32_t head = 0xFACE;
    uint8_t seq = 0;
    uint8_t request = 0;
    uint32_t data = 0;
};
#pragma pack()
enum SelfCheckAck
{
    NOT_CHECK,
    CHECK_OK,
    CHECK_FAILED
};

typedef  std::function<void(const MessageBuffer&)> GetMessageCallback;

class Transport;

class Message
{
public:
    Message(std::function<int32_t(const MessageBuffer&)> sendptr);

    ~Message();
    void SetMessage(const MessageBuffer& sendmsg);
    void GetMessage(const MessageBuffer& recvmsg);
    void RegisterCallback(GetMessageCallback& callback);
    int32_t Init();
    void DeInit();

private:
    void MessageOperate();
    void DispatchThread();

private:
    std::queue<MessageBuffer> message_send_buffer;
    std::queue<MessageBuffer> message_recv_buffer;
    std::function<int32_t(const MessageBuffer&)> sendptr_;
    Transport* transport_;
    std::mutex recv_mutexs_;
    std::condition_variable recv_cvs_;
    std::mutex send_mutexs_;
    std::condition_variable send_cvs_;
    std::shared_ptr<SelfCheck> selfcheck_;
    std::shared_ptr<QuestAndAnswer> qa_;
    std::thread t1_;
    std::thread t2_;
    bool stop_flag = false;
};

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon

#endif // EXTWDG_MESSAGE_H_