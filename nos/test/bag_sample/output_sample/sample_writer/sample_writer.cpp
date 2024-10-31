#include <signal.h>
#include <iostream>
#include "reader.h"
#include "writer.h"

#include <fstream>

#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "proto/soc/sensor_image.pb.h"
#include "proto/test/soc/for_test.pb.h"

using namespace hozon::netaos::bag;

int main(int argc, char** argv) {
    if (argc == 2) {

        Writer writer;
        WriterOptions mcapWriterOptions;
        mcapWriterOptions.output_file_name = "test11";
        writer.Open(mcapWriterOptions);
        // writer.WriteEventProtoMessage<hozon::soc::CompressedImage>(message_vec.topic, proto_data, message_vec.time);

        // std::string path = argv[1];
        // Reader reader;
        // reader.Open(path, "mcap");  //path为.mcap包路经

        // auto topics_info = reader.GetAllTopicsAndTypes();  //获取包中所有的topic和type

        // for (auto topic : topics_info) {
        //     std::cout << topic.first << "; " << topic.second << std::endl;
        // }

        // std::vector<std::string> topics;
        // topics.push_back("/soc/encoded_camera_0");
        // reader.SetFilter(topics);  //设在过滤器，只读取指定的topic的message

        // // 取出一个message的序列化数据后，反序列化为指定的proto格式
        // if (reader.HasNext()) {
        //     TopicMessage message_vec = reader.ReadNext();
        //     std::cout << "message_vec=" << message_vec.topic << "; size=" << message_vec.data.size() << std::endl;
        //     hozon::soc::CompressedImage proto_data = reader.DeserializeToProto<hozon::soc::CompressedImage>(message_vec);
        //     std::cout << "proto_data.length()=" << proto_data.data().size() << std::endl;

        //     Writer writer;
        //     WriterOptions mcapWriterOptions;
        //     mcapWriterOptions.output_file_name = "test";
        //     writer.Open(mcapWriterOptions);
        //     writer.WriteEventProtoMessage<hozon::soc::CompressedImage>(message_vec.topic, proto_data, message_vec.time);
        // }
        return 0;
    }
}
