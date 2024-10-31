/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
 * Description: Document value object class header
 * Create: 2019-6-25
 */

#ifndef ARA_GODEL_COMMON_JSONPARSER_DOCUMENT_H
#define ARA_GODEL_COMMON_JSONPARSER_DOCUMENT_H

#include <vector>
#include <string>
#include <map>
#include <cstdint>
#include <memory>
#include "global.h"

namespace ara {
namespace godel {
namespace common {
namespace jsonParser {
class Document {
public:
    Document() noexcept;
    explicit Document(std::nullptr_t doc) noexcept;
    ~Document();
    Document(Document const &) = default;
    Document(Document &&) = default;
    Document &operator = (Document const &) & = default;
    Document &operator = (Document &&) & = default;
    JsonParseValue Parse(std::string const & path, CRCVerificationType const type = CRCVerificationType::NOT);
    JsonType GetType() const noexcept;
    bool HasParseError() const noexcept;
    Document operator[](int32_t index) const noexcept;
    Document operator[](std::string const & key) const;
    bool HasMember(std::string const & key) const;
    bool IsBool() const noexcept;
    bool GetBool() const noexcept;
    bool IsNull() const noexcept;
    bool IsNumber() const noexcept;
    double GetNumber() const noexcept;
    bool IsString() const noexcept;
    std::string GetString() const;
    bool IsArray() const noexcept;
    std::vector<Document> GetArray() const;
    bool IsObject() const noexcept;
    std::map<std::string, Document> GetObject() const noexcept;
    bool IsUint() const noexcept;
    unsigned GetUint() const noexcept;
    bool IsInt() const noexcept;
    int32_t GetInt() const noexcept;
    bool IsUint8() const noexcept;
    uint8_t GetUint8() const noexcept;
    bool IsInt8() const noexcept;
    int8_t GetInt8() const noexcept;
    bool IsUint16() const noexcept;
    uint16_t GetUint16() const noexcept;
    bool IsInt16() const noexcept;
    int16_t GetInt16() const noexcept;
    bool IsUint32() const noexcept;
    uint32_t GetUint32() const noexcept;
    bool IsInt32() const noexcept;
    int32_t GetInt32() const noexcept;
    bool IsUint64() const noexcept;
    uint64_t GetUint64() const noexcept;
    bool IsInt64() const noexcept;
    int64_t GetInt64() const noexcept;
    bool IsDouble() const noexcept;
    double GetDouble() const noexcept;
    static std::string ParseFileToString(std::string const & path);
    JsonParseValue ParseStringToDocument(std::string const & jsonStr);

    void SetType(JsonType const & type) noexcept;
    void SetNumber(double const & num) noexcept;
    void SetString(std::string const & str);
    void AddVectorElement(Document const & doc);
    void MapEmplace(Document &doc);

private:
    double m_num { 0.0 };
    std::string m_str { "" };
    std::vector<Document> m_vec {};
    std::map<std::string, Document> m_map {};
    std::shared_ptr<Document> m_doc { nullptr };
    JsonType m_type { JsonType::JSON_DEFAULT };

    bool IsInteger() const noexcept;
    bool IsUnsignedInteger() const noexcept;
};
} // namespace jsonParser
} // namespace common
} // namespace godel
} // namespace ara
#endif // ARA_GODEL_COMMON_JSONPARSER_DOCUMENT_H_
