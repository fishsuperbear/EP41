#include "impl/stat_impl.h"
#include <dirent.h>
#include <signal.h>
#include <sys/stat.h>
#include "data_tools_logger.hpp"
#include "google/protobuf/message.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "proto_factory.h"
#include "proto_utility.h"
#include "reader.h"

// #include "proto/dreamview/point_cloud.pb.h"
// #include "proto/perception/perception_obstacle.pb.h"
// #include "proto/soc/sensor_image.pb.h"  // proto 数据变量

namespace hozon {
namespace netaos {
namespace bag {

StatImplmpl::StatImplmpl(/* args */) {}

StatImplmpl::~StatImplmpl() {}

StatErrorCode StatImplmpl::Start(const StatImplOptions& check_option) {
    if (check_option.topics_list.size() != 1) {
        std::cout << "please specified one topic name." << std::endl;
        BAG_LOG_ERROR << "please specified one topic name.";
        return StatErrorCode::K_too_many_topics;
    }

    std::string dirpath = check_option.url;

    // 获取文件信息
    struct stat fileStat;
    if (stat(check_option.url.c_str(), &fileStat) == 0) {
        // 检查文件是否以".map"结尾，并且是普通文件
        if (!S_ISREG(fileStat.st_mode)) {
            BAG_LOG_ERROR << check_option.url << " don't exist.";
            std::cout << check_option.url << " don't exist." << std::endl;
            return StatErrorCode::K_failed;
        }
    } else {
        std::cout << check_option.url << " 获取文件信息失败." << std::endl;
        BAG_LOG_ERROR << check_option.url << " 获取文件信息失败.";
    }

    if (strstr(check_option.url.c_str(), ".mcap") == NULL) {
        std::cout << check_option.url << " ison't a mcap file." << std::endl;
        BAG_LOG_ERROR << check_option.url << " ison't a mcap file.";
        return StatErrorCode::K_failed;
    }

    Reader reader;
    reader.Open(check_option.url, "mcap");             //path为.mcap包路经
    auto topics_info = reader.GetAllTopicsAndTypes();  //获取包中所有的topic和cm type

    for (auto topic_name : check_option.topics_list) {
        // 判断是否找到
        if (topics_info.find(topic_name) != topics_info.end()) {
            //有topic
            std::vector<std::string> topics;
            topics.push_back(topic_name);
            reader.SetFilter(topics);  //设置过滤器，只读取指定的topic的message
            int last_id = -1;
            double last_publish_stamp = 0;

            bool is_first_message = true;
            //直接取出一个message并返回id
            while (reader.HasNext()) {
                int this_id = -1;
                double this_publish_stamp = 0;
                hozon::netaos::bag::TopicMessage topic_message = reader.ReadNext();
                if ("CmProtoBuf" == topics_info[topic_name]) {
                    CmProtoBuf cm_proto_buf = reader.DeserializeToIdl<CmProtoBufPubSubType, CmProtoBuf>(topic_message);
                    std::shared_ptr<google::protobuf::Message> proto_data(hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(cm_proto_buf.name()));
                    if (nullptr == proto_data) {
                        std::cout << "get proto:" << cm_proto_buf.name() << " type faild";
                        return StatErrorCode::K_failed;
                    }
                    proto_data->ParseFromArray(cm_proto_buf.str().data(), cm_proto_buf.str().size());

                    if (hozon::netaos::data_tool_common::ProtoUtility::HasProtoField(*proto_data, "header")) {

                        const google::protobuf::Message& header_msg = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionMessage(*proto_data, "header");
                        this_publish_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(header_msg, "publish_stamp");
                        // double data_stamp = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionDouble(header_msg, "data_stamp");
                        this_id = hozon::netaos::data_tool_common::ProtoUtility::GetProtoReflectionInt32(header_msg, "seq");
                        // double sensor_stamp = 0.0;
                    }

                    // const google::protobuf::Descriptor* proto_data_descriptor = proto_data->GetDescriptor();

                    // // 获取 header 字段描述符
                    // const google::protobuf::FieldDescriptor* header_field = proto_data_descriptor->FindFieldByName("header");
                    // // 获取 header 动态消息
                    // google::protobuf::Message* header_dynamic_message = proto_data->GetReflection()->MutableMessage(proto_data.get(), header_field);
                    // // 获取 header 消息描述符
                    // const google::protobuf::Descriptor* head_descriptor = header_dynamic_message->GetDescriptor();
                    // // 读取 publish_stamp 字段的值
                    // this_id = header_dynamic_message->GetReflection()->GetInt32(*header_dynamic_message, head_descriptor->FindFieldByName("seq"));

                    if (this_id >= 0) {
                        if (is_first_message) {
                            last_id = this_id;
                            last_publish_stamp = this_publish_stamp;
                            is_first_message = false;
                            std::cout << "begin id = " << last_id << ", publish_stamp = " << std::fixed << std::setprecision(9) << last_publish_stamp << std::endl;
                        } else {
                            if ((last_id + 1) != this_id) {
                                std::cout << "\033[31m"
                                          << "Error: lost " << this_id - last_id - 1 << " message: this_message_id=" << last_id << ", publish_stamp = " << std::fixed << std::setprecision(9)
                                          << last_publish_stamp << "; next_message_id=" << this_id << ", publish_stamp = " << std::fixed << std::setprecision(9) << this_publish_stamp << "\033[0m"
                                          << std::endl;
                            }
                            last_id = this_id;
                            last_publish_stamp = this_publish_stamp;
                        }
                    } else {
                        std::cout << " get seq id error" << std::endl;
                        return StatErrorCode::K_failed;
                    }
                }
            }
            std::cout << "end id=" << last_id << ", publish_stamp = " << std::fixed << std::setprecision(9) << last_publish_stamp << std::endl;

        } else {
            std::cout << "no topic:" << topic_name << " in bag" << std::endl;
        }
    }
    return StatErrorCode::K_success;
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon