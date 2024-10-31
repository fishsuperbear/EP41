#include "monitor/cyber_topology_message.h"

#include <unistd.h>
#include <iomanip>
#include <iostream>

#include "data_tools_logger.hpp"
#include "monitor/general_channel_message.h"
#include "monitor/screen.h"

namespace hozon {
namespace netaos {
namespace topic {
constexpr int SecondColumnOffset = 4;

CyberTopologyMessage::CyberTopologyMessage(const std::string& channel, std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager)
    : RenderableMessage(nullptr, 1),
      second_column_(SecondColumnType::MessageFrameRatio),
      pid_(getpid()),
      col1_width_(8),
      specified_channel_(channel),
      all_channels_map_(),
      topic_manager_(topic_manager) {}

CyberTopologyMessage::~CyberTopologyMessage(void) {
    for (auto item : all_channels_map_) {
        if (!GeneralChannelMessage::IsErrorCode(item.second)) {
            delete item.second;
        }
    }
    all_channels_map_.clear();
}

RenderableMessage* CyberTopologyMessage::Child(int line_no) const {
    RenderableMessage* ret = nullptr;
    auto iter = FindChild(line_no);
    if (iter != all_channels_map_.cend() && !GeneralChannelMessage::IsErrorCode(iter->second) && iter->second->is_enabled()) {
        ret = iter->second;
    }
    return ret;
}

std::map<std::string, GeneralChannelMessage*>::const_iterator CyberTopologyMessage::FindChild(int line_no) const {
    --line_no;

    std::map<std::string, GeneralChannelMessage*>::const_iterator ret = all_channels_map_.cend();

    if (line_no > -1 && line_no < page_item_count_) {
        int i = 0;

        auto iter = all_channels_map_.cbegin();
        while (i < page_index_ * page_item_count_) {
            ++iter;
            ++i;
        }

        for (i = 0; iter != all_channels_map_.cend(); ++iter) {
            if (i == line_no) {
                ret = iter;
                break;
            }
            ++i;
        }
    }
    return ret;
}

void CyberTopologyMessage::TopologyChanged(const TopicInfo& changeMsg) {
    // if ("CmProtoBuf" != changeMsg.typeName && "ProtoMethodBase" != changeMsg.typeName && "CmSomeipBuf" != changeMsg.typeName) {
    //     return;
    // }
    if (OPT_JOIN == changeMsg.operate_type) {
        AddReaderWriter(changeMsg.topicName, changeMsg.typeName);
    } else {
        // auto iter = all_channels_map_.find(changeMsg.role_attr().channel_name());

        // if (iter != all_channels_map_.cend() && !GeneralChannelMessage::IsErrorCode(iter->second)) {
        //     // const std::string& node_name = changeMsg.role_attr().node_name();
        //     if (::apollo::cyber::proto::RoleType::ROLE_WRITER == changeMsg.role_type()) {
        //         // iter->second->del_writer(node_name);
        //     } else {
        //         // iter->second->del_reader(node_name);
        //     }
        // }
    }
}

void CyberTopologyMessage::AddReaderWriter(const std::string topic_name, const std::string msgTypeName) {
    if (!specified_channel_.empty() && specified_channel_ != topic_name) {
        return;
    }
    // std::lock_guard<std::mutex> g(cyber_inner_lock_);
    if (static_cast<int>(topic_name.length()) > col1_width_) {
        col1_width_ = static_cast<int>(topic_name.length());
    }

    GeneralChannelMessage* channel_msg = nullptr;
    auto iter = all_channels_map_.find(topic_name);
    if (iter == all_channels_map_.cend()) {
        all_channels_map_[topic_name] = GeneralChannelMessage::CastErrorCode2Ptr(GeneralChannelMessage::ErrorCode::NewSubClassFailed);
        channel_msg = new GeneralChannelMessage(this);

        if (channel_msg != nullptr) {
            channel_msg->set_topic_name(topic_name);
            channel_msg->set_message_type(msgTypeName);
            GeneralChannelMessage* return_code = channel_msg->OpenChannel(topic_name);
            if (GeneralChannelMessage::ErrorCode::TopicAlreadyExist == GeneralChannelMessage::CastPtr2ErrorCode(return_code)) {
                delete channel_msg;
                return;
            } else if (GeneralChannelMessage::IsErrorCode(return_code)) {
                delete channel_msg;
                all_channels_map_[topic_name] = return_code;
            } else {
                all_channels_map_[topic_name] = return_code;
            }
        }
    }
}

void CyberTopologyMessage::ChangeState(const Screen* s, int key) {
    switch (key) {
        case 'f':
        case 'F':
            second_column_ = SecondColumnType::MessageFrameRatio;
            break;

        case 't':
        case 'T':
            second_column_ = SecondColumnType::ProtoType;
            break;

        case ' ': {
            auto iter = FindChild(*line_no());
            if (!GeneralChannelMessage::IsErrorCode(iter->second)) {
                GeneralChannelMessage* child = iter->second;
                if (child->is_enabled()) {
                    child->CloseChannel();
                } else {
                    GeneralChannelMessage* ret = child->OpenChannel(iter->first);
                    if (GeneralChannelMessage::IsErrorCode(ret)) {
                        delete child;
                        all_channels_map_[iter->first] = ret;
                    }
                }
            }
        }

        default: {
        }
    }
}

int CyberTopologyMessage::Render(const Screen* s, int key) {
    page_item_count_ = s->Height() - 1;
    pages_ = static_cast<int>(all_channels_map_.size()) / page_item_count_ + 1;
    ChangeState(s, key);
    SplitPages(key);

    s->AddStr(0, 0, Screen::WHITE_BLACK, "Channels");
    switch (second_column_) {
        case SecondColumnType::ProtoType:
            s->AddStr(col1_width_ + SecondColumnOffset, 0, Screen::WHITE_BLACK, "TypeName");
            break;
        case SecondColumnType::MessageFrameRatio:
            s->AddStr(col1_width_ + SecondColumnOffset, 0, Screen::WHITE_BLACK, "FrameRatio");
            break;
    }

    auto iter = all_channels_map_.cbegin();
    int tmp = page_index_ * page_item_count_;
    int line = 0;
    while (line < tmp) {
        ++iter;
        ++line;
    }

    Screen::ColorPair color;
    std::ostringstream out_str;

    tmp = page_item_count_ + 1;
    for (line = 1; iter != all_channels_map_.cend() && line < tmp; ++iter, ++line) {
        color = Screen::RED_BLACK;

        if (!GeneralChannelMessage::IsErrorCode(iter->second)) {
            if (iter->second->has_message_come()) {
                if (iter->second->is_enabled()) {
                    color = Screen::GREEN_BLACK;
                } else {
                    color = Screen::YELLOW_BLACK;
                }
            }
        }

        s->SetCurrentColor(color);
        s->AddStr(0, line, iter->first.c_str());

        if (!GeneralChannelMessage::IsErrorCode(iter->second)) {
            switch (second_column_) {
                case SecondColumnType::ProtoType:
                    s->AddStr(col1_width_ + SecondColumnOffset, line, iter->second->proto_type().c_str());
                    break;
                case SecondColumnType::MessageFrameRatio: {
                    out_str.str("");
                    out_str << std::fixed << std::setprecision(FrameRatio_Precision) << iter->second->frame_ratio();
                    s->AddStr(col1_width_ + SecondColumnOffset, line, out_str.str().c_str());
                } break;
            }
        } else {
            GeneralChannelMessage::ErrorCode errcode = GeneralChannelMessage::CastPtr2ErrorCode(iter->second);
            s->AddStr(col1_width_ + SecondColumnOffset, line, GeneralChannelMessage::ErrCode2Str(errcode));
        }
        s->ClearCurrentColor();
    }

    return line;
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon