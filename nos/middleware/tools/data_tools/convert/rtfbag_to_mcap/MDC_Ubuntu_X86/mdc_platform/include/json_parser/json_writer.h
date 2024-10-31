/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 * Description: for generate json struct into a string
 * Create: 2019-6-25
 */

#ifndef JSON_WRITER_H
#define JSON_WRITER_H
#include<iostream>
#include<string>
#include<stack>
#include<cstdint>
#include<map>
namespace ara {
namespace godel {
namespace common {
namespace jsonParser {
enum class JsonContainerType: uint8_t {
    DEFAULT_TYPE = 0U,
    OBJECT_TYPE,
    ARRAY_TYPE
};

class JsonWriter {
public:
    JsonWriter() = default;
    ~JsonWriter() = default;
    std::string GetString() const noexcept;
    void StartWriteJson(JsonContainerType type) noexcept;
    void WriteKeyAndValue(std::string const &key, std::string const &value) noexcept;
    void WriteKey(std::string const &key) noexcept;
    void WriteValue(std::string const &value) noexcept;
    void WriteValue(bool value) noexcept;
    void WriteValue(int64_t value) noexcept;
    void WriteValue(uint64_t value) noexcept;
    void WriteValue(double value) noexcept;
    void WriteRawData(std::string const &value) noexcept;
    void EndArray() noexcept;
    void EndObject() noexcept;
private:
void EndContainer() noexcept;
void WriteNumber(std::string const &number) noexcept;
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : " \"" means left pad of string
std::string const leftPad {" \""};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : "\"" means right pad of string
std::string const rightPad {"\""};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : " {" means begin pad of object
std::string const objectPad {" {"};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 :  "} " means end pad of object
std::string const objectEndPad {"} "};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : " [" means begin pad of array
std::string const arrayPad {" ["};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : "] " means end pad of array
std::string const arrayEndPad {"] "};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : ":" means colon pad
std::string const colonPad {":"};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : "," means comma pad
std::string const commaPad {","};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : "true" means true string
std::string const trueString {"true"};
// AXIVION MAGIC AutosarC++19_03-A5.1.1 : "false" means false string
std::string const falseString {"false"};

std::string jsonString_;
uint8_t depth {0U};
std::map<uint8_t, bool> commaPadFlagMap_;
std::stack<JsonContainerType> jsonTypeContainer;
};
}
}
}
}
#endif
