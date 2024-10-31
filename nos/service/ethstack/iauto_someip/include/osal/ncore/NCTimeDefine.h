/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file NCTimeDefine.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTIMEDEFINE_H_
#define INCLUDE_NCORE_NCTIMEDEFINE_H_

/* include of QNX */
#include <osal/ncore/NCTypesDefine.h>
#include <sys/stat.h>
#include <sys/time.h>

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
/* redefinition of QNX */
typedef struct _SYSTEMTIME {
    UINT16 wYear;
    UINT16 wMonth;
    UINT16 wDayOfWeek;
    UINT16 wDay;
    UINT16 wHour;
    UINT16 wMinute;
    UINT16 wSecond;
    UINT16 wMilliseconds;
} SYSTEMTIME;

typedef SYSTEMTIME *PSYSTEMTIME;

/* BCD Time Format                                                       */
typedef struct _BCDTIME {
    UINT8 u8Year;    // The year. The valid values for this member are
                     // 0 through 99
    UINT8 u8Month;   // The month. This member can be one of the
                     // following values. (1-January ~ 12-December)
    UINT8 u8Day;     // The day of the month. The valid values for
                     // this member are 1 through 31.
    UINT8 u8Hour;    // The hour. The valid values for this member are
                     // 0 through 23.
    UINT8 u8Minute;  // The minute. The valid values for this member
                     // are 0 through 59
    UINT8 u8Second;  // The second. The valid values for this member
                     // are 0 through 59
} BCDTIME;

typedef BCDTIME *PBCDTIME;

struct NCTimeStruct {
    UINT16 wYear;          // The year. The valid values for this member are
                           // 1601 through 30827.
    UINT16 wMonth;         // The month. This member can be one of the
                           // following values. (1-January ~ 12-December)
    UINT16 wDayOfWeek;     // The day of the week. This member can be one of
                           // the following values. (0-Sunday ~ 6-Saturday)
    UINT16 wDay;           // The day of the month. The valid values for
                           // this member are 1 through 31.
    UINT16 wHour;          // The hour. The valid values for this member are
                           // 0 through 23.
    UINT16 wMinute;        // The minute. The valid values for this member are
                           // 0 through 59.
    UINT16 wSecond;        // The second. The valid values for this member are
                           // 0 through 59.
    UINT16 wMilliseconds;  // The millisecond. The valid values for this
                           // member are 0 through 999.
};

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set struct timespec with system time since it's start
 * then plus extra msec.
 *
 * @param time: time struct to store time data.
 * @param msec: extra time data.
 */
extern __attribute__( ( visibility( "default" ) ) ) void gettimespec( struct timespec *const t_time,
                                                                      UINT32                 msec );

/**
 * Set struct timespec with msec.
 *
 * @param time: time struct to store time data.
 * @param msec: time data.
 */
extern __attribute__( ( visibility( "default" ) ) ) void settimespec( struct timespec *t_time,
                                                                      UINT32           msec );

/**
 * The substruction of two timespec.
 *
 * @param time1: 1st struct time.
 * @param time2: 2nd struct time.
 * @return  the ms diffenence between time1 and time2.
 */
extern __attribute__( ( visibility( "default" ) ) ) INT64 subtimespec(
    const struct timespec *const time1, const struct timespec *const time2 );

/**
 * Get systemtime to struct systime.
 *
 * @param systime: to store system time data.
 */
extern __attribute__( ( visibility( "default" ) ) ) void GetSystemTime( SYSTEMTIME *systime );

/**
 * Set time with struct systime.
 *
 * @param systime: time data.
 * @return Whether the function call succeed. NC_TRUE indicates success.
    Otherwise indicates failure.
 */
extern __attribute__( ( visibility( "default" ) ) ) NC_BOOL SetSystemTime(
    const SYSTEMTIME *const systime );

/**
 * Get local time to struct systime.
 *
 * @param systime: to store system time data.
 */
extern __attribute__( ( visibility( "default" ) ) ) void GetLocalTime( SYSTEMTIME *systime );

/**
 * Set local time with struct systime.
 *
 * @param systime: time data.
 * @return Whether the function call succeed. NC_TRUE indicates success.
    Otherwise indicates failure.
 */
extern __attribute__( ( visibility( "default" ) ) ) NC_BOOL SetLocalTime(
    const SYSTEMTIME *const systime );

/**
 * Get system total running time with ms unit.
 *
 * @return system total running time since it's start.
 */
extern __attribute__( ( visibility( "default" ) ) ) INT64 GetTickCount();

/**
 * Get thread total running time with ms unit.
 *
 * @return thread total running since it's start.
 */
extern __attribute__( ( visibility( "default" ) ) ) INT64 GetThreadTime();

/**
 * Check if the borken time is in the valid scope.
 *
 * @param systime: time data to be check.
 * @return Whether the function call succeed. NC_TRUE indicates success.
    Otherwise indicates failure.
 */
extern __attribute__( ( visibility( "default" ) ) ) NC_BOOL isValidSystemTime(
    const SYSTEMTIME *const systime );

/**
 * Convert SystemTime to BCDTime data format.
 *
 * @param bcdtime: store new time data.
 * @param systime: src time data.
 * @return Whether the function call succeed. NC_TRUE indicates success.
    Otherwise indicates failure.
 * @note: If the conversion was failed, the result would be undefined.
 */
extern __attribute__( ( visibility( "default" ) ) ) NC_BOOL SystemTimeToBCDTime(
    BCDTIME *bcdtime, const SYSTEMTIME *const systime );

/**
 * Convert BCDTime to SystemTime data format.
 *
 * @param systime: store new time data.
 * @param bcdtime: src time data.
 * @return Whether the function call succeed. NC_TRUE indicates success.
    Otherwise indicates failure.
 * @note: If the conversion was failed, the result would be undefined.
 */
extern __attribute__( ( visibility( "default" ) ) ) NC_BOOL BCDTimeToSystemTime(
    SYSTEMTIME *systime, const BCDTIME *const bcdtime );
#ifdef __cplusplus
}
#endif
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCTIMEDEFINE_H_
/* EOF */
