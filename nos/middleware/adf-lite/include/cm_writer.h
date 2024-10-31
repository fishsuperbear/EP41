#pragma once
#include "cm/include/skeleton.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "adf-lite/include/adf_lite_internal_logger.h"

using namespace hozon::netaos::cm;
namespace hozon {
namespace netaos {
namespace adf_lite {

class CMWriter {
public:
    CMWriter() : 
        _skeleton(std::make_shared<CmProtoBufPubSubType>()) {
    }

    ~CMWriter() {}

    int32_t Init(const uint32_t domain, const std::string& topic) {
        return _skeleton.Init(domain, topic);
    }

    void Deinit() {
        _skeleton.Deinit();
    }

    int32_t Write(const std::string& topic_type, const std::string &data) {
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);

        if (cm_pb == nullptr) {
            ADF_INTERNAL_LOG_ERROR << "cm_pb pointer is nullptr!!!";
            return -1;
        }
        cm_pb->name(topic_type);
        cm_pb->str().assign(data.begin(), data.end());

        return _skeleton.Write(cm_pb);
    }

    int32_t Write(const std::shared_ptr<google::protobuf::Message> data) {
        if (data == nullptr) {
            ADF_INTERNAL_LOG_ERROR << "data pointer is nullptr!!!";
            return -1;
        }
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);

        if (cm_pb == nullptr) {
            ADF_INTERNAL_LOG_ERROR << "cm_pb pointer is nullptr!!!";
            return -1;
        }
        cm_pb->name(data->GetTypeName());
        std::string serialized_data;
        data->SerializeToString(&serialized_data);
        cm_pb->str().assign(serialized_data.begin(), serialized_data.end());
        return _skeleton.Write(cm_pb);
    }
private:
    Skeleton _skeleton;
};

}
}
}