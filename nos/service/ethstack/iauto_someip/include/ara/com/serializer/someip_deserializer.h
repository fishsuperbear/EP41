#ifndef SOMEIP_DESERIALIZE_H_
#define SOMEIP_DESERIALIZE_H_

#include <string>
#include "ara/core/string.h"
#include "ara/com/serializer/transformation_props.h"

struct NESomeIPExtendAttr;
class NESomeIPPayloadDeserializerGeneral;

class SomeIPDeSerializer
{
public:
    using DeployType = void;
    static constexpr bool TlvEnable = false;

    SomeIPDeSerializer(const ara::com::SomeipTransformationProps* deployment = nullptr);
    ~SomeIPDeSerializer();

    int readValue(int8_t& value);
    int readValue(int16_t& value);
    int readValue(int32_t& value);
    int readValue(int64_t& value);
    int readValue(uint8_t& value);
    int readValue(uint16_t& value);
    int readValue(uint32_t& value);
    int readValue(uint64_t& value);
    int readValue(bool& value);
    int readValue(float& value);
    int readValue(double& value);
    int readValue(std::string& value, uint32_t size = 0);
    int readValue(ara::core::String& value, uint32_t size = 0) ;

    int readValue(uint8_t** data, uint32_t* size);
    int readValue(uint8_t* data, uint32_t size);
    int readValue(std::vector<uint8_t>& data, uint32_t size);
    int readValue(std::vector<int8_t>& data, uint32_t size);

    int readVectorBegin();
    int readVectorEnd();

    int readArrayBegin();
    int readArrayEnd();

    int readStructBegin();
    int readStructEnd();

    int readUnionBegin(uint32_t& type);
    int readUnionEnd();

    int readTlvBegin(uint32_t tlv_id);
    int readTlvEnd();

    int lastError();

    int from_buffer(const std::vector<uint8_t>& buffer);

private:
    NESomeIPExtendAttr* m_deployment;
    NESomeIPPayloadDeserializerGeneral* m_deserializer;
};

#endif // SOMEIP_DESERIALIZE_H_
/* EOF */
