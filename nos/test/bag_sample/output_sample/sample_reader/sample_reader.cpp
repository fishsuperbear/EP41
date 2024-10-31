
#include <signal.h>

#include <fstream>
#include <iostream>

#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "reader.h"

#include "proto/test/soc/for_test.pb.h"

using namespace hozon::netaos::bag;

int main(int argc, char** argv) {
    if (argc == 2) {
        std::string path = argv[1];
        Reader reader;
        reader.Open(path, "mcap");  // path为.mcap包路经

        auto topics_info = reader.GetAllTopicsAndTypes();  //获取包中所有的topic和type

        std::vector<std::string> topics;
        topics.push_back("/soc/encoded_camera_0");
        reader.SetFilter(topics);  //设在过滤器，只读取指定的topic的message

        //取出一个message的序列化数据后，反序列化为指定的proto格式
        if (reader.HasNext()) {
            TopicMessage message_vec = reader.ReadNext();
            std::cout << "message_vec=" << message_vec.topic << "; size=" << message_vec.data.size() << std::endl;
            adf::lite::dbg::WorkflowResult proto_data = reader.DeserializeToProto<adf::lite::dbg::WorkflowResult>(message_vec);
            std::cout << "proto_data.length()=" << proto_data.mutable_header()->frame_id() << std::endl;
        }

        //直接取出一个反序列化为指定的proto格式的message
        if (reader.HasNext()) {
            // hozon::perception::CompressedImage proto_data = reader.ReadNextProtoMessage<hozon::perception::CompressedImage>();
            adf::lite::dbg::WorkflowResult proto_data = reader.ReadNextProtoMessage<adf::lite::dbg::WorkflowResult>();
            std::cout << "proto_data.length()=" << proto_data.mutable_header()->frame_id() << std::endl;
        }

        //直接取出一个反序列化为指定的fast dds idl格式的message
        if (reader.HasNext()) {
            CmProtoBuf cm_proto_buf = reader.ReadNextIdlMessage<CmProtoBufPubSubType, CmProtoBuf>();
            std::cout << "cm_proto_buf.name()=" << cm_proto_buf.name() << ";  cm_proto_buf.str().size() =" << cm_proto_buf.str().size() << std::endl;
        }

        //取出一个message,并以json形式返回
        int cout = 0;
        while (reader.HasNext() && cout < 10) {
            // hozon::perception::CompressedImage proto_data = reader.ReadNextProtoMessage<hozon::perception::CompressedImage>();
            BinaryType binary_type;
            std::vector<uint8_t> binary_data;
            std::string json_str;
            std::string topic_name;
            int64_t typetime;
            reader.ReadNextAsJson(typetime, topic_name, json_str, binary_type, binary_data);
            // std::cout << "proto_data json =" << json_str << std::endl;
            std::cout << "typetime =" << typetime << std::endl;
            std::cout << "topic_name =" << topic_name << std::endl;
            std::cout << "binary_type=" << binary_type << std::endl;
            std::cout << "binary_data=" << binary_data.size() << std::endl;
            std::ofstream ofs("./test" + std::to_string(cout) + ".jpg", std::ios::binary | std::ios::out);
            ofs.write(std::string(binary_data.begin(), binary_data.end()).c_str(), binary_data.size());
            cout++;
        }
    }
    return 0;
}