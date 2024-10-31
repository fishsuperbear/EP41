#ifndef _SAMPLE_STATE_COMMON_H_
#define _SAMPLE_STATE_COMMON_H_


enum class STANDBY_STATE : uint8_t {
    STANDBY_DEFAULT = 0,            // default
    STANDBY_FAPA_PARKING_IN = 1,    // fapa parking in
    STANDBY_FAPA_PARKING_OUT = 2,   // fapa parking out
    STANDBY_FAPA_PARKING_SELECT = 3,// fapa parking select
    STANDBY_RPA_PARKING_IN = 4,     // rpa parking in
    STANDBY_RPA_PARKING_OUT = 5,    // rpa parking out
    STANDBY_RPA_PARKING_SELECT = 6, // rpa parking select
    STANDBY_RPA_PARKING_LINE = 7,   // rpa parking line
};

enum class PARKING_STATE : uint8_t {
    PARKING = 0,                    // default
    FAPA_PARKING_IN = 1,            // fapa parking in
    FAPA_PARKING_OUT = 2,           // fapa parking out
    RPA_PARKING_IN = 3,             // rpa parking in
    RPA_PARKING_OUT = 4,            // rpa parking out
    RPA_PARKING_SELECT = 5,         // rpa parking select
    RPA_PARKING_LINE = 6,           // rpa parking line
    TBA_PARKING = 7,                // tba
};

#endif