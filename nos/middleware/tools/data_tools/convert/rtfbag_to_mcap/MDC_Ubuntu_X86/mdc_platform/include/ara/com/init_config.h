/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 * Description: Init authenticator opts for SecOC api
 * Create: 2022-02-18
 */
#ifndef ARA_COM_SECOC_INIT_CONFIG_H
#define ARA_COM_SECOC_INIT_CONFIG_H

#include <set>
#include <functional>
#include <memory>
#include <cstdint>
#include <securec.h>
#include <ara/core/vector.h>
#include "ara/hwcommon/log/log.h"

namespace ara {
namespace com {
enum class SecOCAuthenticatorGenType : std::uint8_t {
    OVERALL = 0U,
    SEGMENT = 1U,
    UNDEFINED = 0xFFU
};

enum class SecOCAuthenticatorGenConfigTag : std::uint8_t {
    KEY_ID = 0
};

enum class SecOCAlgoType : std::uint8_t {
    SECOC_AES_CMAC_128 = 0,
    SECOC_SIP_HASH_2_4
};

class SecOCAuthenticatorGenConfig final {
public:
    SecOCAuthenticatorGenConfig(
        SecOCAuthenticatorGenConfigTag tag,
        const uint8_t* tagData,
        uint16_t tagDataLength);
    ~SecOCAuthenticatorGenConfig() = default;
    SecOCAuthenticatorGenConfig(const SecOCAuthenticatorGenConfig& other);
    SecOCAuthenticatorGenConfig& operator=(const SecOCAuthenticatorGenConfig& other);
    SecOCAuthenticatorGenConfig(SecOCAuthenticatorGenConfig&& other);
    SecOCAuthenticatorGenConfig& operator=(SecOCAuthenticatorGenConfig&& other);
    SecOCAuthenticatorGenConfigTag GetSecOCAuthenticatorGenConfigTag() const noexcept
    {
        return tag_;
    }
    const uint8_t* GetTagData() const noexcept
    {
        return tagData_;
    }
    uint16_t GetTagDataLength() const noexcept
    {
        return tagDataLength_;
    }
private:
    static constexpr std::size_t TAG_DATA_MAX_LENGTH_BYTE = 32U;
    SecOCAuthenticatorGenConfigTag tag_;
    uint8_t tagData_[TAG_DATA_MAX_LENGTH_BYTE];
    uint16_t tagDataLength_;    // byte
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
};

enum class GenResult : std::uint8_t {
    SUCCESS = 0,
    FAILED,
    BUSY
};

class SecOCAuthenticatorGenResult final {
public:
    SecOCAuthenticatorGenResult(GenResult result, const std::shared_ptr<ara::core::vector<uint8_t>> generatedData)
     : result_(result), generatedData_(generatedData), generatorHandler_(nullptr) {}
    SecOCAuthenticatorGenResult(GenResult result, void* generatorHandler)
     : result_(result), generatedData_(nullptr), generatorHandler_(generatorHandler) {}
    ~SecOCAuthenticatorGenResult() = default;
    SecOCAuthenticatorGenResult(const SecOCAuthenticatorGenResult& other) = default;
    SecOCAuthenticatorGenResult& operator=(const SecOCAuthenticatorGenResult& other) = delete;
    SecOCAuthenticatorGenResult(SecOCAuthenticatorGenResult&& other) = default;
    SecOCAuthenticatorGenResult& operator=(SecOCAuthenticatorGenResult&& other) = delete;
    GenResult GetGenResult() const noexcept
    {
        return result_;
    }
    const std::shared_ptr<ara::core::vector<uint8_t>> GetGeneratedData() const noexcept
    {
        return generatedData_;
    }
    void* GetGeneratedHandler() const noexcept
    {
        return generatorHandler_;
    }
private:
    GenResult result_{GenResult::SUCCESS};
    const std::shared_ptr<ara::core::vector<uint8_t>> generatedData_;
    void* generatorHandler_;
};

class SecOCAlgoConfigType final {
public:
    SecOCAlgoConfigType(SecOCAlgoType algoType, const uint8_t* keyIdBuff, uint8_t keyIdBuffLength)
     : algoType_(algoType), keyIdBuff_(keyIdBuff), keyIdBuffLength_(keyIdBuffLength) {}
    ~SecOCAlgoConfigType() = default;
    SecOCAlgoConfigType(const SecOCAlgoConfigType& other) = default;
    SecOCAlgoConfigType& operator=(const SecOCAlgoConfigType& other) = default;
    SecOCAlgoConfigType(SecOCAlgoConfigType&& other) = default;
    SecOCAlgoConfigType& operator=(SecOCAlgoConfigType&& other) = default;
    SecOCAlgoType GetAlgoType() const noexcept
    {
        return algoType_;
    }
    const uint8_t* GetKeyIdData() const noexcept
    {
        return keyIdBuff_;
    }
    uint16_t GetKeyIdLength() const noexcept
    {
        return keyIdBuffLength_;
    }
private:
    SecOCAlgoType algoType_;
    const uint8_t* keyIdBuff_;
    uint8_t keyIdBuffLength_;   // byte
};

class CalculateDataBuff final {
public:
    CalculateDataBuff(const uint8_t* dataBuff, uint32_t dataLength)
     : dataBuff_(dataBuff), dataLength_(dataLength) {}
    ~CalculateDataBuff() = default;
    CalculateDataBuff(const CalculateDataBuff& other) = default;
    CalculateDataBuff& operator=(const CalculateDataBuff& other) = default;
    CalculateDataBuff(CalculateDataBuff&& other) = default;
    CalculateDataBuff& operator=(CalculateDataBuff&& other) = default;
    const uint8_t* GetCalculateData() const noexcept
    {
        return dataBuff_;
    }
    uint32_t GetDataLength() const noexcept
    {
        return dataLength_;
    }
private:
    const uint8_t* dataBuff_;
    uint32_t dataLength_;   // byte
};

using InitSecOCAuthenticatorGenerator =
    std::function<bool(const ara::core::vector<SecOCAuthenticatorGenConfig>& initConfig)>;
using GenerateSecOCAuthenticator =
    std::function<SecOCAuthenticatorGenResult(const SecOCAlgoConfigType& algoConfig, const CalculateDataBuff& calculateData)>;
using GenerateSecOCAuthenticatorBegin =
    std::function<SecOCAuthenticatorGenResult(const SecOCAlgoConfigType& algoConfig, const CalculateDataBuff& calculateData)>;
using GenerateSecOCAuthenticatorUpdate =
    std::function<SecOCAuthenticatorGenResult(void* const genHandler, const CalculateDataBuff& calculateData)>;
using GenerateSecOCAuthenticatorFinish =
    std::function<SecOCAuthenticatorGenResult(void* const genHandler, const CalculateDataBuff& calculateData)>;

class InitConfig final {
public:
    static bool SetSecOCAuthenticatorGenType(SecOCAuthenticatorGenType type) noexcept;
    static SecOCAuthenticatorGenType GetSecOCAuthenticatorGenType() noexcept;
    static bool RegisterInitSecOCAuthenticatorGenerator(
        const std::set<SecOCAuthenticatorGenConfigTag>& initConfig, const InitSecOCAuthenticatorGenerator& func) noexcept;
    static const std::set<SecOCAuthenticatorGenConfigTag>& GetInitSecOCConfigType() noexcept;
    static InitSecOCAuthenticatorGenerator GetInitSecOCAuthenticatorGenerator() noexcept;
    static bool RegisterGenerateSecOCAuthenticator(const GenerateSecOCAuthenticator& func) noexcept;
    static GenerateSecOCAuthenticator GetGenerateSecOCAuthenticator() noexcept;
    static bool RegisterGenerateSecOCAuthenticatorBegin(const GenerateSecOCAuthenticatorBegin& func) noexcept;
    static GenerateSecOCAuthenticatorBegin GetGenerateSecOCAuthenticatorBegin() noexcept;
    static bool RegisterGenerateSecOCAuthenticatorUpdate(const GenerateSecOCAuthenticatorUpdate& func) noexcept;
    static GenerateSecOCAuthenticatorUpdate GetGenerateSecOCAuthenticatorUpdate() noexcept;
    static bool RegisterGenerateSecOCAuthenticatorFinish(const GenerateSecOCAuthenticatorFinish& func) noexcept;
    static GenerateSecOCAuthenticatorFinish GetGenerateSecOCAuthenticatorFinish() noexcept;
private:
    static SecOCAuthenticatorGenType genType_;
    static std::set<SecOCAuthenticatorGenConfigTag> configTag_;
    static InitSecOCAuthenticatorGenerator initSecOCAuthenticatorGenerator_;
    static GenerateSecOCAuthenticator generateSecOCAuthenticator_;
    static GenerateSecOCAuthenticatorBegin generateSecOCAuthenticatorBegin_;
    static GenerateSecOCAuthenticatorUpdate generateSecOCAuthenticatorUpdate_;
    static GenerateSecOCAuthenticatorFinish generateSecOCAuthenticatorFinish_;
};
}
}
#endif  // ARA_COM_INIT_CONFIG_H
