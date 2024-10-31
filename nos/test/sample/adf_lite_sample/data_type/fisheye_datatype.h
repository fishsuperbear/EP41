#pragma once
#include <map>
#include "serialize.h"
#include "adf-lite/include/base.h"
 //#include "adf-lite/include/struct_register.h"
namespace hozon {
namespace netaos {
namespace adf_lite {

struct TestStruct : BaseData {
    bool isValid;
    std::vector<int32_t> info;
};

// 对于扁平的struct，不要重写SerializeAsString, ParseFromString
class TestPlainStruct : public BaseData {
public:
    int32_t data;
    int32_t data2;
};

class TestNotPlainStruct : public BaseData {
public:
    int32_t data;
    int32_t data2;
    std::string data_str;
    std::vector<int32_t> data_vec;
    std::map<int64_t, std::string> data_map;

    uint32_t GetSerializeSize() {
        uint32_t total_size = sizeof(BaseData)  + sizeof(int32_t) + sizeof(int32_t) + sizeof(uint32_t) + data_str.size();
        total_size += sizeof(uint32_t) + sizeof(int32_t) * data_vec.size();
        // map的大小
        total_size += sizeof(uint32_t);
        for (auto it = data_map.begin(); it != data_map.end(); it++) {
            total_size += sizeof(int64_t) + sizeof(uint32_t) + it->second.size();
        }
        return total_size;
    }

    virtual void SerializeAsString(std::string &s_data, std::size_t size) {
        BaseData::SerializeAsString(s_data, sizeof(BaseData)); //序列化基类内容

        s_data.resize(1000);

        /* 也可以精确计算size，定义string, 比如：
        s_data.resize(GetSerializeSize()); */

        uint32_t offset = sizeof(BaseData);

        serialize_data(data, s_data, offset);
        serialize_data(data2, s_data, offset);
        serialize_string(data_str, s_data, offset);
        serialize_vector<int32_t>(data_vec, s_data, offset);

        uint32_t map_start_offset = offset;
        offset += sizeof(uint32_t);
        for (auto it = data_map.begin(); it != data_map.end(); it++) {
            //序列化key
            serialize_data((it->first), s_data, offset);
            //序列化value
            serialize_string((it->second), s_data, offset);
        }
        //将offset写入到map_start_offset位置
        serialize_data_pure<uint32_t>(offset, s_data, map_start_offset);

    }

    virtual void ParseFromString(std::string &s_data, std::size_t size) {
        BaseData::ParseFromString(s_data, sizeof(BaseData)); //序列化基类内容

        uint32_t offset = sizeof(BaseData);
        unserialize_data<int32_t>(data, s_data, offset);
        unserialize_data<int32_t>(data2, s_data, offset);
        unserialize_string(data_str, s_data, offset);
        unserialize_vector<int32_t>(data_vec, s_data, offset);

        //读取map数据截止位置
        uint32_t map_end_offset;
        unserialize_data<uint32_t>(map_end_offset, s_data, offset);

        // 反序列化map
        while (offset < map_end_offset) {
            int64_t map_key;
            unserialize_data<int64_t>(map_key, s_data, offset);
            std::string value;
            unserialize_string(value, s_data, offset);
            data_map[map_key] = value;
        }
    }
};

// 也可以使用宏来序列化，反序列化
class TestNotPlainStruct2 : public BaseData {
public:
    int32_t data;
    int32_t data2;
    std::string data_str;
    std::vector<int32_t> data_vec;
    std::map<int64_t, std::string> data_map;

    virtual void SerializeAsString(std::string &s_data, std::size_t size) {
        BaseData::SerializeAsString(s_data, sizeof(BaseData)); //序列化基类内容

        s_data.resize(1000);

        /* 也可以精确计算size，定义string, 比如：
        s_data.resize(GetSerializeSize()); */

        uint32_t offset = sizeof(BaseData);

        SERIALIZE_DATA(data, s_data, offset);
        SERIALIZE_DATA(data2, s_data, offset);
        SERIALIZE_STRING(data_str, s_data, offset);
        SERIALIZE_VECTOR(data_vec, s_data, offset);

        uint32_t map_start_offset = offset;
        offset += sizeof(uint32_t);
        for (auto it = data_map.begin(); it != data_map.end(); it++) {
            //序列化key
            SERIALIZE_DATA((it->first), s_data, offset)
            //序列化value
            SERIALIZE_STRING((it->second), s_data, offset)
        }
        //将offset写入到map_start_offset位置
        SERIALIZE_DATA_PURE(offset, s_data, map_start_offset)
    }

    virtual void ParseFromString(std::string &s_data, std::size_t size) {
        BaseData::ParseFromString(s_data, sizeof(BaseData)); //序列化基类内容

        uint32_t offset = sizeof(BaseData);
        UNSERIALIZE_DATA(int32_t, data, s_data, offset)
        UNSERIALIZE_DATA(int32_t, data2, s_data, offset)
        UNSERIALIZE_STRING(data_str, s_data, offset);
        UNSERIALIZE_VECTOR(data_vec, uint32_t, s_data, offset);

        //读取map数据截止位置
        uint32_t map_end_offset;
        UNSERIALIZE_DATA(uint32_t, map_end_offset, s_data, offset)

        // 反序列化map
        while (offset < map_end_offset) {
            int64_t map_key;
            UNSERIALIZE_DATA(int64_t, map_key, s_data, offset)
            std::string value;
            UNSERIALIZE_STRING(value, s_data, offset)
            data_map[map_key] = value;
        }
    }
};

}
}
}
