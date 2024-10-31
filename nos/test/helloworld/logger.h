#include <iostream>

namespace hozon {
namespace netaos {
namespace log {

enum class LogLevel : uint8_t {
    kOff = 0x00U,
    kFatal = 0x01U,
    kError = 0x02U,
    kWarn = 0x03U,
    kInfo = 0x04U,
    kDebug = 0x05U,
    kVerbose = 0x06U
};

class EasyLogStream {
public:
    EasyLogStream(LogLevel config_level, LogLevel statement_level) {
        _valid = (config_level >= statement_level);
    }
    
    EasyLogStream(const EasyLogStream &) = delete;
    EasyLogStream &operator =(const EasyLogStream &) = delete;
    EasyLogStream &operator =(EasyLogStream &&) = delete;

    EasyLogStream(EasyLogStream&& another) {
        another._newline = false;
        _newline = true;
        _valid = another._valid;
    }

    ~EasyLogStream() {
        if (_newline && _valid) {
            std::cout << std::endl;
        }
    }

    template<typename T>
    EasyLogStream& operator<<(const T& value) {
        if (_valid) {
            std::cout << value;
        }

        return *this;
    }

private:
    bool _valid = false;
    bool _newline = true;
};

class EmptyEasyStream : public EasyLogStream {
    template<typename T>
    EmptyEasyStream& operator<<(const T& value) {
        return *this;
    }
};

class EasyLogger {
public:
    void Init(LogLevel level) {
        _level = level;
    }

    EasyLogStream LogFatal() {
        return std::move(EasyLogStream(_level, LogLevel::kFatal) << "[Fatal] ");
    }

    EasyLogStream LogError() {
        return std::move(EasyLogStream(_level, LogLevel::kError) << "[Error] ");
    }

    EasyLogStream LogWarn() {
        return std::move(EasyLogStream(_level, LogLevel::kWarn) << "[Warn] ");
    }

    EasyLogStream LogInfo() {
        return std::move(EasyLogStream(_level, LogLevel::kInfo) << "[Info] ");
    }

    EasyLogStream LogDebug() {
        return std::move(EasyLogStream(_level, LogLevel::kDebug) << "[Debug] ");
    }

    EasyLogStream LogVerbose() {
        return std::move(EasyLogStream(_level, LogLevel::kVerbose) << "[Verbose] ");
    }

private:
    LogLevel _level = LogLevel::kInfo;
};

}    
}    
}