#pragma once

#include "cm/include/skeleton.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"

namespace hozon {
namespace netaos {
namespace cm {

template<typename MessageT>
class ProtoCMWriter {
public:
    ProtoCMWriter() : 
            _skeleton(std::make_shared<CmProtoBufPubSubType>()) {

    }

    ~ProtoCMWriter() {

    }

    int32_t Init(const uint32_t domain, const std::string& topic) {
        return _skeleton.Init(domain, topic);
    }

    void Deinit() {
        _skeleton.Deinit();
    }

    int32_t Write(std::shared_ptr<MessageT> data) {
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);

        cm_pb->name(data->GetTypeName());
        std::string serialized_data;
        data->SerializeToString(&serialized_data);
        cm_pb->str().assign(serialized_data.begin(), serialized_data.end());

        return _skeleton.Write(cm_pb);
    }

    int32_t Write(const MessageT& data) {
        std::shared_ptr<CmProtoBuf> cm_pb(new CmProtoBuf);

        cm_pb->name(data.GetTypeName());
        std::string serialized_data;
        data.SerializeToString(&serialized_data);
        cm_pb->str().assign(serialized_data.begin(), serialized_data.end());

        return _skeleton.Write(cm_pb);
    }

private:
    Skeleton _skeleton;
};

}
}
}