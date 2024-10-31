#ifndef FAULT_MESSAGE_PARSER_H
#define FAULT_MESSAGE_PARSER_H

#include <cstdint>
#include <memory>
#include <ctime>
#include <mutex>

#include "yaml-cpp/yaml.h"

#include "common/logger.h"



namespace hozon {
namespace ethstack {
namespace lidar {


#pragma pack(push, 1)

struct PreHeader {
    uint8_t leader1;
    uint8_t leader2;

};

struct TimeDivisionMultiplexing {
    uint8_t flag;
    uint8_t data[26];
};

struct FaultMessage {
    PreHeader header;
    uint8_t fault_message_version;
    uint8_t data_time[6];
    uint32_t timestamp;
    uint8_t lidar_operation_state;
    uint8_t lidar_fault_state;
    uint8_t fault_code_type;
    uint8_t rolling_counter;
    uint8_t total_fault_code_number;
    uint8_t fault_code_ID;
    uint32_t fault_code;
    TimeDivisionMultiplexing time_division_multiplexing;
    uint8_t reserve[12];
    uint8_t fault_indication;
    uint32_t crc;
    uint8_t cyber_security[32];
};

#pragma pack(pop)



class FaultMessageParser {
   public:
    static FaultMessageParser& Instance();
    virtual ~FaultMessageParser();

    void Parse(uint8_t* dataptr, uint32_t size);


   private:
    FaultMessageParser();
    uint8_t CurrentStatus_;

};

}  // namespace lidar
}  // namespace ethstack
}  // namespace hozon
#endif  // FAULT_MESSAGE_PARSER_H