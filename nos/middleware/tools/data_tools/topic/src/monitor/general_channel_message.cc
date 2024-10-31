#include "monitor/general_channel_message.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/subscriber/SampleInfo.h>
#include "cm/include/cm_config.h"
#include "data_tools_logger.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"
#include "idl/generated/zerocopy_imageTypeObject.h"
#include "monitor/general_message.h"
#include "monitor/screen.h"
#include "proto_factory.h"
#include "impl/someip_deserialize_impl.h"

namespace hozon {
namespace netaos {
namespace topic {

using namespace hozon::netaos::data_tool_common;

static constexpr size_t kGB = 1 << 30;
static constexpr size_t kMB = 1 << 20;
static constexpr size_t kKB = 1 << 10;

const char* GeneralChannelMessage::ErrCode2Str(GeneralChannelMessage::ErrorCode errCode) {
    const char* ret;
    switch (errCode) {

        case GeneralChannelMessage::ErrorCode::CreateReaderFailed:
            ret = "Cannot Create Reader";
            break;

        case GeneralChannelMessage::ErrorCode::CreatTopicFailed:
            ret = "Cannot Create Topic";
            break;

        case GeneralChannelMessage::ErrorCode::TopicAlreadyExist:
            ret = "Topic Already Subscribe";
            break;

        case GeneralChannelMessage::ErrorCode::TopicManagerFailed:
            ret = "TopicManager init Failed";
            break;

        case GeneralChannelMessage::ErrorCode::NewSubClassFailed:
            ret = "New SubClass Failed";

        default:
            ret = "Unknown Error Code";
    }
    return ret;
}

bool GeneralChannelMessage::IsErrorCode(void* ptr) {
    GeneralChannelMessage::ErrorCode err = (GeneralChannelMessage::ErrorCode)(reinterpret_cast<intptr_t>(ptr));
    switch (err) {
        case ErrorCode::CreateReaderFailed:
        case ErrorCode::CreatTopicFailed:
        case ErrorCode::TopicAlreadyExist:
        case ErrorCode::TopicManagerFailed:
        case ErrorCode::NewSubClassFailed:
            return true;

        default: {
        }
    }
    return false;
}

double GeneralChannelMessage::frame_ratio(void) {
    if (!is_enabled() || !has_message_come()) {
        return 0.0;
    }
    std::chrono::steady_clock::time_point time_now = std::chrono::steady_clock::now();
    auto interval = time_now - time_last_calc_;
    if (std::chrono::duration_cast<std::chrono::nanoseconds>(interval).count() > 1000000000) {
        int old = frame_counter_;
        while (!frame_counter_.compare_exchange_strong(old, 0)) {}
        if (old == 0) {
            return 0.0;
        }
        auto curMsgTime = msg_time_;
        auto deltaTime = curMsgTime - last_time_;
        frame_ratio_ = old / std::chrono::duration_cast<std::chrono::duration<double>>(deltaTime).count();  //转换成秒
        last_time_ = curMsgTime;
        time_last_calc_ = time_now;
    }
    return frame_ratio_;
}

void GeneralChannelMessage::MyListener::on_data_available(eprosima::fastdds::dds::DataReader* reader) {
    eprosima::fastdds::dds::SampleInfo info;
    if ("CmProtoBuf" == parents_->topic_->get_type_name()) {
        CmProtoBuf temp_cmProtoBuf;
        if (reader->take_next_sample(&(temp_cmProtoBuf), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                parents_->set_has_message_come(true);
                parents_->msg_time_ = std::chrono::steady_clock::now();
                parents_->proto_name_ = temp_cmProtoBuf.name();
                ++parents_->frame_counter_;
                std::lock_guard<std::mutex> _g(parents_->inner_lock_);
                parents_->topic_message_ = temp_cmProtoBuf.str();
            }
        }
    } else if ("CmSomeipBuf" == parents_->topic_->get_type_name()) {
        CmSomeipBuf temp_cmSomeipBuf;
        if (reader->take_next_sample(&(temp_cmSomeipBuf), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                parents_->set_has_message_come(true);
                parents_->msg_time_ = std::chrono::steady_clock::now();
                parents_->proto_name_ = temp_cmSomeipBuf.name();
                ++parents_->frame_counter_;
                std::lock_guard<std::mutex> _g(parents_->inner_lock_);
                parents_->topic_message_ = temp_cmSomeipBuf.str();
            }
        }
    } else if ("ProtoMethodBase" == parents_->topic_->get_type_name()) {
        ProtoMethodBase protoMethod_data;
        if (reader->take_next_sample(&(protoMethod_data), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                parents_->set_has_message_come(true);
                parents_->msg_time_ = std::chrono::steady_clock::now();
                parents_->proto_name_ = protoMethod_data.name();
                ++parents_->frame_counter_;
                std::lock_guard<std::mutex> _g(parents_->inner_lock_);
                parents_->topic_message_ = protoMethod_data.str();
            }
        }
    } else if (parents_->topic_->get_type_name().find("ZeroCopyImg8M420") != std::string::npos) {
        FASTDDS_SEQUENCE(ZeroCopyImg8M420Seq, ZeroCopyImg8M420);
        ZeroCopyImg8M420Seq data;
        SampleInfoSeq infos;
        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg8M420& sample = data[i];
                }
            }
            reader->return_loan(data, infos);
        }

        parents_->set_has_message_come(true);
        parents_->msg_time_ = std::chrono::steady_clock::now();
        parents_->proto_name_ = "unknow";
        ++parents_->frame_counter_;
        std::lock_guard<std::mutex> _g(parents_->inner_lock_);
        // parents_->topic_message_ = "";
    } else if (parents_->topic_->get_type_name().find("ZeroCopyImg2M422") != std::string::npos) {
        FASTDDS_SEQUENCE(ZeroCopyImg2M422Seq, ZeroCopyImg2M422);
        ZeroCopyImg2M422Seq data;
        SampleInfoSeq infos;
        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                // Check whether the DataSample contains data or is only used to communicate of a
                // change in the instance
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg2M422& sample = data[i];
                }
            }
            reader->return_loan(data, infos);
        }
        parents_->set_has_message_come(true);
        parents_->msg_time_ = std::chrono::steady_clock::now();
        parents_->proto_name_ = "unknow";
        ++parents_->frame_counter_;
        std::lock_guard<std::mutex> _g(parents_->inner_lock_);
        // parents_->topic_message_ = "";
    } else if (parents_->topic_->get_type_name().find("ZeroCopyImg3M422") != std::string::npos) {
        FASTDDS_SEQUENCE(ZeroCopyImg3M422Seq, ZeroCopyImg3M422);
        ZeroCopyImg3M422Seq data;
        SampleInfoSeq infos;
        if (ReturnCode_t::RETCODE_OK == reader->take(data, infos)) {
            // Iterate over each LoanableCollection in the SampleInfo sequence
            for (LoanableCollection::size_type i = 0; i < infos.length(); ++i) {
                if (infos[i].valid_data) {
                    // Print the data.
                    const ZeroCopyImg3M422& sample = data[i];
                }
            }
            reader->return_loan(data, infos);
        }
        parents_->set_has_message_come(true);
        parents_->msg_time_ = std::chrono::steady_clock::now();
        parents_->proto_name_ = "unknow";
        ++parents_->frame_counter_;
        std::lock_guard<std::mutex> _g(parents_->inner_lock_);
        // parents_->topic_message_ = "";
    } else {
        TOPIC_LOG_ERROR << parents_->topic_->get_type_name() << " unsurpport. Can be analyze.";
    }
};

void GeneralChannelMessage::MyListener::on_subscription_matched(DataReader* reader, const SubscriptionMatchedStatus& info) {
    parents_->set_has_message_come(false);
}

GeneralChannelMessage* GeneralChannelMessage::OpenChannel(const std::string& topic_name) {
    if (topic_name.empty()) {
        TOPIC_LOG_ERROR << "topic_name is empty";
    }

    int32_t dds_data_type = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetTopicDataType(topic_name_);
    auto participant_ptr = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetParticipant(dds_data_type, false);
    auto subscriber_ptr = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetSubscriber(dds_data_type);
    if (nullptr == participant_ptr || nullptr == subscriber_ptr) {
        return CastErrorCode2Ptr(ErrorCode::TopicManagerFailed);
    }

    if (topic_ != nullptr && nullptr == reader_) {
        eprosima::fastdds::dds::DataReaderQos qos_ = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetReaderQos(dds_data_type, topic_name_);
        eprosima::fastdds::dds::StatusMask sub_mask = eprosima::fastdds::dds::StatusMask::subscription_matched() << eprosima::fastdds::dds::StatusMask::data_available();
        reader_ = subscriber_ptr->create_datareader(topic_, qos_, &reader_listener_, sub_mask);
        if (nullptr == reader_) {
            return CastErrorCode2Ptr(ErrorCode::CreateReaderFailed);
        }
        return this;
    }

    //订阅
    topic_ = participant_ptr->find_topic(topic_name, eprosima::fastrtps::Duration_t(0, 5));
    if (nullptr != topic_) {
        //多个线程同时为相同的topic的create_datareader时，程序可能会崩溃
        return CastErrorCode2Ptr(ErrorCode::TopicAlreadyExist);
    } else {
        topic_ = participant_ptr->create_topic(topic_name, message_type_, TOPIC_QOS_DEFAULT);
    }
    if (topic_ == nullptr) {
        return CastErrorCode2Ptr(ErrorCode::CreatTopicFailed);
    }
    eprosima::fastdds::dds::DataReaderQos qos_ = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetReaderQos(dds_data_type, topic_name_);
    eprosima::fastdds::dds::StatusMask sub_mask = eprosima::fastdds::dds::StatusMask::subscription_matched() << eprosima::fastdds::dds::StatusMask::data_available();
    reader_ = subscriber_ptr->create_datareader(topic_, qos_, &reader_listener_, sub_mask);
    if (nullptr == reader_) {
        return CastErrorCode2Ptr(ErrorCode::CreateReaderFailed);
    }
    BAG_LOG_DEBUG << "[monitor] subscribe topic:" << topic_name << " type:" << message_type_;
    return this;
}

int GeneralChannelMessage::Render(const Screen* s, int key) {
    switch (key) {
        case 'b':
        case 'B':
            current_state_ = State::ShowDebugString;
            break;

            // case 'i':
            // case 'I':
            //     current_state_ = State::ShowInfo;
            //     break;

        default: {
        }
    }

    clear();

    int line_no = 0;

    s->SetCurrentColor(Screen::WHITE_BLACK);
    s->AddStr(0, line_no++, "TopicName: ");
    s->AddStr(topic_name().c_str());

    s->AddStr(0, line_no++, "ProtoType: ");
    s->AddStr(proto_type().c_str());

    if (is_enabled()) {
        if (State::ShowDebugString == current_state_) {
            RenderDebugString(s, key, &line_no);
        }

        // switch () {
        //     case State::ShowDebugString:
        //         RenderDebugString(s, key, &line_no);
        //         break;
        //         // case State::ShowInfo:
        //         //     RenderInfo(s, key, &line_no);
        //         //     break;
        // }
    } else {
        s->AddStr(0, line_no++, "Channel has been closed");
    }
    s->ClearCurrentColor();

    return line_no;
}

void GeneralChannelMessage::RenderDebugString(const Screen* s, int key, int* line_no) {
    if (has_message_come()) {
        // if (raw_msg_class_ == nullptr) {
        //     raw_msg_class_ = ProtoFactory::getInstance()->GenerateMessageByType(proto_type());
        // }

        std::string prot_type = proto_type();
        std::string someip_head = "/someip";
        if (prot_type.compare(0, someip_head.length(), someip_head) == 0) {
            // 说明是someip 数据
            s->AddStr(0, (*line_no)++, "someip payload deserialized: ");
            std::vector<char> topic_message = CopyMsg();
            if (topic_message.size()) {
                std::string debugstr = hozon::netaos::someip_deserialize::SomeipDeserializeImpl::getInstance()->deserialize(topic_message.data(), topic_message.size(), prot_type);
                // s->AddStr(0, (*line_no)++, debugstr.c_str());
                // 下面是按monitor的要求输出
                std::vector<std::string> lines;
                std::stringstream ss(debugstr);
                std::string line;
                while (std::getline(ss, line)) {
                    lines.push_back(line);
                }

                int lcount = lines.size();
                page_item_count_ = s->Height() - *line_no;
                pages_ = lcount / page_item_count_ + 1;
                SplitPages(key);
                int jump_lines = page_index_ * page_item_count_;
                PrintSomeipMessage(this, lines, jump_lines, s, line_no, 0);
            }
        } else if (raw_msg_class_ == nullptr) {
            raw_msg_class_ = ProtoFactory::getInstance()->GenerateMessageByType(proto_type());
            s->AddStr(0, (*line_no)++, "Cannot Generate Message by Message Type");
        } else {
            s->AddStr(0, (*line_no)++, "FrameRatio: ");

            std::ostringstream out_str;
            out_str << std::fixed << std::setprecision(FrameRatio_Precision) << frame_ratio();
            s->AddStr(out_str.str().c_str());

            std::vector<char> topic_message = CopyMsg();
            // decltype(topic_message_) channel_msg;

            if (topic_message.size()) {
                s->AddStr(0, (*line_no)++, "RawMessage Size: ");
                out_str.str("");
                out_str << topic_message.size() << " Bytes";
                if (topic_message.size() >= kGB) {
                    out_str << " (" << static_cast<float>(topic_message.size()) / kGB << " GB)";
                } else if (topic_message.size() >= kMB) {
                    out_str << " (" << static_cast<float>(topic_message.size()) / kMB << " MB)";
                } else if (topic_message.size() >= kKB) {
                    out_str << " (" << static_cast<float>(topic_message.size()) / kKB << " KB)";
                }
                s->AddStr(out_str.str().c_str());
                if (raw_msg_class_->ParseFromArray(topic_message.data(), topic_message.size())) {
                    int lcount = LineCount(*raw_msg_class_, s->Width());
                    page_item_count_ = s->Height() - *line_no;
                    pages_ = lcount / page_item_count_ + 1;
                    SplitPages(key);
                    int jump_lines = page_index_ * page_item_count_;
                    jump_lines <<= 2;
                    jump_lines /= 5;
                    GeneralMessageBase::PrintMessage(this, *raw_msg_class_, &jump_lines, s, line_no, 0);
                } else {
                    s->AddStr(0, (*line_no)++, "Cannot parse the raw message");
                }
            } else {
                s->AddStr(0, (*line_no)++, "The size of this raw Message is Zero");
            }
        }
    } else {
        s->AddStr(0, (*line_no)++, "No Message Came");
    }
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon