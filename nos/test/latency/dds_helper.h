#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>

class DebugLogger {
public:
    DebugLogger(bool need_log) :
        _need_log(need_log) {

    }

    ~DebugLogger() {
        if (_need_log) {
            std::cout << std::endl;
        }
    }

    template<typename T>
    DebugLogger& operator<<(const T& value) {
        if (_need_log) {
            std::cout << _head << value;
        }
        
        return *this;
    }

private:
    bool _need_log = false;
    std::string _head;
};

#define LAT_LOG_FATAL DebugLogger(true) << "[Fatal] "
#define LAT_LOG_ERROR DebugLogger(true) << "[Error] "
#define LAT_LOG_WARN DebugLogger(true) << "[Warn] "
#define LAT_LOG_INFO DebugLogger(true) << "[Info] "
#define LAT_LOG_DEBUG DebugLogger(true) << "[Debug] "
#define LAT_LOG_VERBOSE DebugLogger(true) << "[Verbose] "

class DDSHelper {
public:
    static DDSHelper& GetInstance(); 

    eprosima::fastdds::dds::DomainParticipant* _participant;
    eprosima::fastdds::dds::Subscriber* _subscriber;
    eprosima::fastdds::dds::Publisher* _publisher;

private:
    DDSHelper();
};
