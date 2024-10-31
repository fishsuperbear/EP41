#ifndef SOMEIP_SERIALIZE_H_
#define SOMEIP_SERIALIZE_H_

#include <string>
#include "ara/core/string.h"
#include "ara/com/serializer/transformation_props.h"

struct NESomeIPExtendAttr;
class NESomeIPPayloadSerializerGeneral;

class SomeIPSerializer
{
public:
    using DeployType = void;
    static constexpr bool TlvEnable = false;

    explicit SomeIPSerializer(const ara::com::SomeipTransformationProps* deployment = nullptr);
    ~SomeIPSerializer();

    int writeValue(const int8_t& value);
    int writeValue(const int16_t& value);
    int writeValue(const int32_t& value);
    int writeValue(const int64_t& value);
    int writeValue(const uint8_t& value);
    int writeValue(const uint16_t& value);
    int writeValue(const uint32_t& value);
    int writeValue(const uint64_t& value);
    int writeValue(const bool& value);
    int writeValue(const float& value);
    int writeValue(const double& value);
    int writeValue(const std::string& value);
    int writeValue(const ara::core::String& value);

    int writeValue(const uint8_t* data, uint32_t size);
    int writeValue(const std::vector<uint8_t>& data);
    int writeValue(const std::vector<int8_t>& data);

    int writeVectorBegin();
    int writeVectorEnd();

    int writeArrayBegin();
    int writeArrayEnd();

    int writeStructBegin();
    int writeStructEnd();

    int writeUnionBegin(uint32_t type);
    int writeUnionEnd();

    int writeTlvBegin(int tlv_id);
    int writeTlvEnd();

    int lastError();
    const std::vector<uint8_t>& getData();

private:
    NESomeIPExtendAttr* m_deployment;
    NESomeIPPayloadSerializerGeneral* m_serializer;
};

#endif // SOMEIP_SERIALIZE_H_
/* EOF */
