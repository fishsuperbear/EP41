#include "reader.h"
#include <signal.h>
#include <iostream>

#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "proto/drivers/sensor_image.pb.h"
#include "proto/soc/for_test.pb.h"

using namespace hozon::netaos::bag;

int main(int argc, char** argv) {
    if (argc == 2) {
        std::string path = argv[1];
        Reader reader;
        reader.Open(path, "mcap");

        std::vector<std::string> topics;
        topics.push_back("workresult_pb");
        reader.SetFilter(topics);

        if (reader.HasNext()) {
            TopicMessage message_vec = reader.ReadNext();
            std::cout << "message_vec=" << message_vec.topic << "; size=" << message_vec.data.size() << std::endl;
            adf::lite::dbg::WorkflowResult proto_data = reader.DeserializeToProto<adf::lite::dbg::WorkflowResult>(message_vec);
            std::cout << "proto_data.length()=" << proto_data.mutable_header()->frame_id() << std::endl;
        }

        if (reader.HasNext()) {
            CmProtoBuf cm_proto_buf = reader.ReadNextIdlMessage<CmProtoBufPubSubType, CmProtoBuf>();
            std::cout << "cm_proto_buf.name()=" << cm_proto_buf.name() << ";  cm_proto_buf.str().size() =" << cm_proto_buf.str().size() << std::endl;
        }

        if (reader.HasNext()) {
            // hozon::perception::CompressedImage proto_data = reader.ReadNextProtoMessage<hozon::perception::CompressedImage>();
            adf::lite::dbg::WorkflowResult proto_data = reader.ReadNextProtoMessage<adf::lite::dbg::WorkflowResult>();
            std::cout << "proto_data.length()=" << proto_data.mutable_header()->frame_id() << std::endl;
        }
    }
    return 0;
}