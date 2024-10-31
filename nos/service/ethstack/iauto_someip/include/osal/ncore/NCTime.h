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
 * @file NCTime.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCTIME_H_
#define INCLUDE_NCORE_NCTIME_H_

#include "osal/ncore/NCTimeDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

// Class declaration
class NCTime;

/**
 * @brief
 *
 * @class NCTimeSpan
 */
class __attribute__( ( visibility( "default" ) ) ) NCTimeSpan {
   public:
    static const INT64  TicksPerMillisecond   = 0x2710LL;
    static const INT64  TicksPerSecond        = 0x989680LL;
    static const INT64  TicksPerMinute        = 0x23c34600LL;
    static const INT64  TicksPerHour          = 0x861c46800LL;
    static const INT64  TicksPerDay           = 0xc92a69c000LL;
    static const INT64  DaysPerYear           = 0x16dLL;  // 0x16d = 365
    static const INT64  MinTicks              = 0x0LL;
    static const INT64  MaxTicks              = 0x2bca2875f4373fffLL;
    static const UINT64 KindLocalAmbiguousDst = 0xC000000000000000ULL;

   public:
    /**
     * Constructor
     *
     * @param d The day of the month. The valid values for this member are -24854
     * through 24854.
     * @param h The hour. The valid values for this member are -23 through 23.
     * @param mi The minute. The valid values for this member are -59 through 59.
     * @param s The second. The valid values for this member are -59 through 59.
     * @param ms The millisecond. The valid values for this member are -999
     * through 999.
     */
    NCTimeSpan( INT32 d, INT32 h, INT32 mi, INT32 s, INT32 ms = 0 );

    /**
     * Copy constructor
     */
    NCTimeSpan( const NCTimeSpan &ts );

    /**
     * operator =
     */
    NCTimeSpan &operator=( const NCTimeSpan &ts );

    /**
     * Destructor
     */
    ~NCTimeSpan();

    /**
     * Set DateTime
     *
     * @param d The day of the month. The valid values for this member are -24854
     * through 24854.
     * @param h The hour. The valid values for this member are -23 through 23.
     * @param mi The minute. The valid values for this member are -59 through 59.
     * @param s The second. The valid values for this member are -59 through 59.
     * @param ms The millisecond. The valid values for this member are -999
     * through 999.
     */
    VOID set( INT32 d, INT32 h, INT32 mi, INT32 s, INT32 ms = 0 );

    /**
     * Get the year which storaged in this class.
     *
     * @return UINT32 year (-24854 ~ 24854)
     */
    INT32 getDay() const;

    /**
     * Get the hour which storaged in this class.
     *
     * @return UINT32 hour (-23 ~ 23)
     */
    INT32 getHour() const;

    /**
     * Get the minute which storaged in this class.
     *
     * @return UINT32 minute (-59 ~ 59)
     */
    INT32 getMinute() const;

    /**
     * Get the second which storaged in this class.
     *
     * @return UINT32 second (-59 ~ 59)
     */
    INT32 getSecond() const;

    /**
     * Get the millisecond which storaged in this class.
     *
     * @return UINT32 millisecond (-999 ~ 999)
     */
    INT32 getMillisecond() const;

    /**
     * operator ==
     */
    NC_BOOL operator==( const NCTimeSpan &ts ) const;

    /**
     * operator !=
     */
    NC_BOOL operator!=( const NCTimeSpan &ts ) const;

    /**
     * operator <
     */
    NC_BOOL operator<( const NCTimeSpan &ts ) const;

    /**
     * operator <=
     */
    NC_BOOL operator<=( const NCTimeSpan &ts ) const;

    /**
     * operator >
     */
    NC_BOOL operator>( const NCTimeSpan &ts ) const;

    /**
     * operator >=
     */
    NC_BOOL operator>=( const NCTimeSpan &ts ) const;

    /**
     * operator +=
     */
    NCTimeSpan &operator+=( const NCTimeSpan &ts );

    /**
     * operator -=
     */
    NCTimeSpan &operator-=( const NCTimeSpan &ts );

   public:
    NCTimeSpan();
    explicit NCTimeSpan( INT64 ticks );
    INT64               getTicks() const;
    static INT64        TimeToTicks( INT32 h, INT32 mi, INT32 s );
    static INT64        DayTimeToTicks( INT32 d, INT32 h, INT32 mi, INT32 s, INT32 ms );
    static INT64        DateToTicks( INT32 y, INT32 m, INT32 d );
    static INT64        NCTimeToTicks( const NCTimeStruct & );
    static NCTimeStruct TicksToNCTime( const INT64 & );
    static INT64        add( const NCTimeStruct &, const NCTimeSpan & );
    static INT64        subtract( const NCTimeStruct &, const NCTimeSpan & );
    static INT64        subtract( const NCTimeStruct &, const NCTimeStruct & );
    static NC_BOOL      isLeapYear( INT32 y );
    static INT32        DayOfWeek( INT32 y, INT32 m, INT32 d );

   private:
    INT64 m_ticks;
};

/**
 * operator for NCTimeSpan = NCTimeSpan + NCTimeSpan
 */
__attribute__( ( visibility( "default" ) ) ) NCTimeSpan operator+( const NCTimeSpan &lhs,
                                                                   const NCTimeSpan &rhs );

/**
 * operator for NCTimeSpan = NCTimeSpan - NCTimeSpan
 */
__attribute__( ( visibility( "default" ) ) ) NCTimeSpan operator-( const NCTimeSpan &lhs,
                                                                   const NCTimeSpan &rhs );

/**
 * @brief
 *
 * @class NCTime
 */
class __attribute__( ( visibility( "default" ) ) ) NCTime {
   public:
    friend NCTimeSpan operator-( const NCTime &lhs, const NCTime &rhs );

    /**
     * Construct an zero date&time
     *
     */
    NCTime();

    /**
     * Construct an appointted date&time
     *
     * @param y The year. The valid values for this member are 1601 through 30827.
     * @param m The month. This member can be one of the following values.
     * (1-January ~ 12-December)
     * @param d The day of the month. The valid values for this member are 1
     * through 31.
     * @param h The hour. The valid values for this member are 0 through 23.
     * @param mi The minute. The valid values for this member are 0 through 59.
     * @param s The second. The valid values for this member are 0 through 59.
     * @param ms The millisecond. The valid values for this member are 0 through
     * 999.
     */
    NCTime( UINT32 y, UINT32 m, UINT32 d, UINT32 h = 0U, UINT32 mi = 0U, UINT32 s = 0U,
            UINT32 ms = 0U );

    /**
     * Copy constructor
     */
    NCTime( const NCTime & );

    /**
     * operator =
     */
    NCTime &operator=( const NCTime & );

    /**
     * Destructor
     *
     */
    ~NCTime();

    /**
     * Get the year of the date&time which storaged in this class.
     *
     * @return UINT32 year of the date&time
     */
    UINT16 getYear() const;

    /**
     * Get the month of the date&time which storaged in this class.
     *
     * @return UINT month of the date&time (1-January ~ 12-December)
     */
    UINT16 getMonth() const;

    /**
     * Get the week of the date&time which storaged in this class.
     *
     * @return UINT week of the date&time (0-Sunday ~ 6-Saturday)
     */
    UINT16 getDayOfWeek() const;

    /**
     * Get the day of the date&time which storaged in this class.
     *
     * @return UINT day of the date&time (1 ~ 31)
     */
    UINT16 getDay() const;

    /**
     * Get the hour of the date&time which storaged in this class.
     *
     * @return UINT hour of the date&time (0 ~ 23)
     */
    UINT16 getHour() const;

    /**
     * Get the minute of the date&time which storaged in this class.
     *
     * @return UINT minute of the date&time (0 ~ 59)
     */
    UINT16 getMinute() const;

    /**
     * Get the second of the date&time which storaged in this class.
     *
     * @return UINT second of the date&time (0 ~ 59)
     */
    UINT16 getSecond() const;

    /**
     * Get the millisecond of the date&time which storaged in this class.
     *
     * @return UINT millisecond of the date&time (0 ~ 999)
     */
    UINT16 getMillisecond() const;

    /**
     * Set the date&time which storaged in this class.
     *
     */
    VOID set( UINT32 y, UINT32 m, UINT32 d, UINT32 h = 0U, UINT32 mi = 0U, UINT32 s = 0U,
              UINT32 ms = 0U );

    /**
     * Set the time which storaged in this class.
     *
     * @param h The hour. The valid values for this member are 0 through 23.
     * @param mi The minute. The valid values for this member are 0 through 59.
     * @param s The second. The valid values for this member are 0 through 59.
     * @param ms The millisecond. The valid values for this member are 0 through
     * 999.
     */
    VOID setTime( UINT32 h, UINT32 mi, UINT32 s, UINT32 ms = 0U );

    /**
     * Set the date which storaged in this class.
     *
     * @param y The year. The valid values for this member are 1601 through 30827.
     * @param m The month. This member can be one of the following values.
     * (1-January ~ 12-December)
     * @param d The day of the month. The valid values for this member are 1
     * through 31.
     */
    VOID setDate( UINT32 y, UINT32 m, UINT32 d );

    /**
     * Storage the Current Local Time.
     *
     */
    VOID getNow();

    /**
     * Storage the Current System Time (UTC).
     *
     */
    VOID getUTCNow();

    /**
     * Get the current cpu tickcount.
     *
     */
    static INT64 getTickCount();

    /**
     * operator == to compare two NCTime whether equal.
     */
    NC_BOOL operator==( const NCTime & ) const;

    /**
     * operator != to compare two NCTime whether not equal.
     */
    NC_BOOL operator!=( const NCTime & ) const;

    /**
     * operator += to add a NCTimeSpan.
     */
    NCTime &operator+=( const NCTimeSpan &ts );

    /**
     * operator -= to subtract a NCTimeSpan.
     */
    NCTime &operator-=( const NCTimeSpan &ts );

    NC_BOOL operator<( const NCTime & ) const;
    NC_BOOL operator<=( const NCTime & ) const;
    NC_BOOL operator>( const NCTime & ) const;
    NC_BOOL operator>=( const NCTime & ) const;

    NC_BOOL isValidTimeDate() const;
    NC_BOOL isValidTime() const;
    NC_BOOL isValidDate() const;

   private:
    VOID         set( SYSTEMTIME & );
    NCTimeStruct m_cTime;
};

/**
 * operator for NCTime = NCTime + NCTimeSpan
 */
__attribute__( ( visibility( "default" ) ) ) NCTime operator+( const NCTime &    lhs,
                                                               const NCTimeSpan &rhs );

/**
 * operator for NCTime = NCTime - NCTimeSpan
 */
__attribute__( ( visibility( "default" ) ) ) NCTime operator-( const NCTime &    lhs,
                                                               const NCTimeSpan &rhs );

/**
 * operator for NCTimeSpan = NCTime - NCTime
 */
__attribute__( ( visibility( "default" ) ) ) NCTimeSpan operator-( const NCTime &lhs,
                                                                   const NCTime &rhs );

OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCTIME_H_
/* EOF */
