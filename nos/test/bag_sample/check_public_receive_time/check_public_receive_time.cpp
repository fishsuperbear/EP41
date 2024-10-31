#include <signal.h>
#include <iostream>
#include "ament_index_cpp/include/ament_index_cpp/get_search_paths.hpp"
#include "data_tools/bag/reader.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "proto_factory/proto_factory.h"
#include "proto/dreamview/point_cloud.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
#include "proto/soc/sensor_image.pb.h"
#include "proto/soc/sensor_image.pb.h"  // proto 数据变量

#include <dirent.h>
#include <sys/stat.h>

using namespace hozon::netaos::bag;

// std::vector<std::string> check_topic = {"/soc/pointcloud"};

bool isdir(std::string& path) {
    struct stat fileStat;
    if (stat(path.c_str(), &fileStat) == 0) {
        if (S_ISREG(fileStat.st_mode)) {
            printf("这是一个文件\n");
            return false;
        } else if (S_ISDIR(fileStat.st_mode)) {
            printf("这是一个目录\n");
            return true;
        } else {
            printf("既不是文件也不是目录\n");
            return false;
        }
    } else {
        printf("获取文件/目录信息失败\n");
        return false;
    }
}

int main(int argc, char** argv) {
    if (argc == 3) {
        std::string dirpath = argv[1];
        std::vector<std::string> check_file;

        if (isdir(dirpath)) {
            std::cout << "ison't .mcap file" << std::endl;
        } else {
            check_file.push_back(dirpath);
        }

        std::vector<std::string> check_topic;

        check_topic.push_back(argv[2]);
        for (auto topic_name : check_topic) {
            for (auto file_path : check_file) {
                std::cout << "topic_name: " << topic_name << std::endl;
                Reader reader;
                reader.Open(file_path, "mcap");  //path为.mcap包路经
                std::cout << "open: " << file_path << std::endl;

                auto topics_info = reader.GetAllTopicsAndTypes();  //获取包中所有的topic和cm type
                // auto it = std::find(topics_info.begin(), topics_info.end(), topic_name);
                // 判断是否找到
                if (topics_info.find(topic_name) != topics_info.end()) {
                    //有topic
                    std::vector<std::string> topics;
                    topics.push_back(topic_name);
                    reader.SetFilter(topics);  //设置过滤器，只读取指定的topic的message
                    int last_id = -1;

                    //直接取出一个message并返回id
                    while (reader.HasNext()) {

                        // CmProtoBuf cm_proto_buf = TopicMessage ReadNext();
                        hozon::netaos::bag::TopicMessage topic_message = reader.ReadNext();
                        CmProtoBuf cm_proto_buf = reader.DeserializeToIdl<CmProtoBufPubSubType, CmProtoBuf>(topic_message);
                        std::shared_ptr<google::protobuf::Message> proto_data(hozon::netaos::data_tool_common::ProtoFactory::getInstance()->GenerateMessageByType(cm_proto_buf.name()));
                        if (nullptr == proto_data) {
                            std::cout << "get proto:" << cm_proto_buf.name() << " type faild";
                            return ReaderErrorCode::FILE_FAILED;
                        }
                        proto_data->ParseFromArray(cm_proto_buf.str().data(), cm_proto_buf.str().size());
                        const google::protobuf::Descriptor* proto_data_descriptor = proto_data->GetDescriptor();

                        // 获取 header 字段描述符
                        const google::protobuf::FieldDescriptor* header_field = proto_data_descriptor->FindFieldByName("header");
                        // 获取 header 动态消息
                        google::protobuf::Message* header_dynamic_message = proto_data->GetReflection()->MutableMessage(proto_data.get(), header_field);
                        // 获取 header 消息描述符
                        const google::protobuf::Descriptor* head_descriptor = header_dynamic_message->GetDescriptor();
                        // 读取 publish_stamp 字段的值
                        double publish_stamp_value = header_dynamic_message->GetReflection()->GetDouble(*header_dynamic_message, head_descriptor->FindFieldByName("publish_stamp"));

                        //演示将字段设置为一些值
                        // proto_data->GetReflection()->SetInt32(proto_data.get(), proto_data->GetDescriptor()->FindFieldByName("your_field_name"), 42);
                        printf("topic_name =: %s, publish %f, receive: %f \n", topic_name.c_str(), publish_stamp_value, (double)topic_message.time / 1000000000);
                    }

                } else {
                    std::cerr << "file_path does not has " << topic_name << std::endl;
                }
            }
        }
        return 0;
    }
}
