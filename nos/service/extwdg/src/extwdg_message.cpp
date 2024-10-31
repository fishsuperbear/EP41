/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: main function definition
 */

#include <chrono>
#include "extwdg_message.h"

namespace hozon {
namespace netaos {
namespace extwdg {

#define PHM_SELFCHECK_REQUEST 0x1
#define PHM_SOC2MCU_REQUEST 0x2
#define QUEST_ANSWER_REQUEST 0x3
#define DEINIT_REQUEST 0x4

Message::Message(std::function<int32_t(const MessageBuffer&)> sendptr)
{
    sendptr_ = sendptr;
    selfcheck_ = std::make_shared<SelfCheck>();
    qa_ = std::make_shared<QuestAndAnswer>();
}

Message::~Message()
{

}

int32_t
Message::Init()
{
    EW_INFO << "Message::Init enter!";
    selfcheck_->Init();
    t2_ = std::thread(&Message::MessageOperate, this);
    t1_ = std::thread(&Message::DispatchThread, this);

    return 0;
}

void
Message::DeInit()
{
    EW_INFO << "Message::DeInit deinit!";
    stop_flag = true;
    selfcheck_->DeInit();
    t1_.join();
    t2_.join();
    EW_INFO << "Message::DeInit success!";
}

void
Message::SetMessage(const MessageBuffer& sendmsg)
{
    std::unique_lock<std::mutex> lck(send_mutexs_);
    message_send_buffer.push(sendmsg);
    send_cvs_.notify_one();
}

void
Message::GetMessage(const MessageBuffer& recvmsg)
{
    EW_INFO << "Message::GetMessage enter!";
    std::unique_lock<std::mutex> lck(recv_mutexs_);
    message_recv_buffer.push(recvmsg);
    recv_cvs_.notify_one();
}

void
Message::DispatchThread()
{
    EW_INFO << "Message::DispatchThread enter!";
    while(!stop_flag) {
        std::unique_lock<std::mutex> lck(send_mutexs_);
        send_cvs_.wait(lck, [this]{return !message_send_buffer.empty();});
        EW_INFO << "Message::DispatchThread loop!";
        while(!message_send_buffer.empty()) {
            MessageBuffer sendmsg = message_send_buffer.front();
            message_send_buffer.pop();
            if(sendmsg.request == DEINIT_REQUEST) {
                return;
            }
            sendptr_(sendmsg);
        }
    }
}

void
Message::MessageOperate()
{
    EW_INFO << "Message::MessageOperate enter!";
    while(!stop_flag) {
        std::unique_lock<std::mutex> lck(recv_mutexs_);
        recv_cvs_.wait(lck, [this]{return !message_recv_buffer.empty();});
        EW_INFO << "Message::MessageOperate message comming!";
        while(!message_recv_buffer.empty()) {
            MessageBuffer message = message_recv_buffer.front();
            message_recv_buffer.pop();
            MessageBuffer sendmsg;
            if(message.request == PHM_SELFCHECK_REQUEST) {
                int32_t res = selfcheck_->RequestSelfCheck();
                if(-1 == res) {
                    SendFault_t fault(4920, 4, 1);
                    PhmClientInstance::getInstance()->ReportFault(fault);
                    sendmsg.data = SelfCheckAck::CHECK_FAILED;
                }
                else if(0 == res) {
                    sendmsg.data = SelfCheckAck::CHECK_OK;
                }
                else {
                    sendmsg.data = SelfCheckAck::NOT_CHECK;
                }
            } 
            else if (message.request == PHM_SOC2MCU_REQUEST) {

                int32_t res = selfcheck_->RequestSoc2Mcu();
                if(-1 == res) {
                    SendFault_t fault(4920, 5, 1);
                    PhmClientInstance::getInstance()->ReportFault(fault);
                    sendmsg.data = SelfCheckAck::CHECK_FAILED;
                }
                else if(0 == res) {
                    sendmsg.data = SelfCheckAck::CHECK_OK;
                }
                else {
                    sendmsg.data = SelfCheckAck::NOT_CHECK;
                }
            } 
            else if (message.request == DEINIT_REQUEST) {
                SetMessage(message);
                
                return;
            }
            else {
                EW_INFO << "Message::MessageOperate quest !";
                uint32_t answer;
                answer = qa_->Quest(message.data);
                sendmsg.data = answer;
            }

            sendmsg.seq = message.seq;
            sendmsg.request = message.request;
            SetMessage(sendmsg);
            EW_INFO << "Message request is  !"<< sendmsg.request;
            EW_INFO << "Message data is  !"<< sendmsg.data;
        }
    }
}

void
Message::RegisterCallback(GetMessageCallback& callback)
{
    EW_INFO << "Message::RegisterCallback enter!";
    callback = std::bind(&Message::GetMessage, this, std::placeholders::_1);
    return;
}

}  // namespace extwdg
}  // namespace netaos
}  // namespace hozon