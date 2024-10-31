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
 * @file NCString.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCSTRING_H_
#define INCLUDE_NCORE_NCSTRING_H_

#include "osal/ncore/NCConverter.h"
#include "osal/ncore/NCString_DEF.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
typedef NC_BOOL ( *NC_Judger )( const CHAR *chVal, UINT16 &chLen );
NC_BOOL NC_JudgeSpace_Backward( const CHAR *const chVal, UINT16 &chLen );
NC_BOOL NC_JudgeSpace_Forward( const CHAR *const chVal, UINT16 &chLen );

/**
 * @brief A String manipulate class
 *
 * A class manage the buffer, codepage, serialization, deserialization and etc
 * of the string.
 * We strongly suggest you to use this class instead of C-style string in your
 * code.
 *
 */
class __attribute__( ( visibility( "default" ) ) ) NCString {
   public:
    /**
     * Construct an non-string. This string even does not have its buffer.
     * It is different from empty string which has a '\\0' in its own buffer.
     */
    NCString();

    /**
     * Copy constructor.
     */
    NCString( const NCString &rhs );

    /**
     * Construct a string with a CHAR array.
     * If NSTRING_UNICODE is not defined, it uses UTF-8 codepage.
     * @param p The CHAR array.
     */
    NCString( const CHAR *const p );

    /**
     * Construct a string with a CHAR array and a specified length.
     * If NSTRING_UNICODE is not defined, it uses UTF-8 codepage.
     * @param p The CHAR array.
     * @param l Length to set to NCString, (not include the terminator).
     */
    NCString( const CHAR *const p, UINT32 l );

#ifndef NSTRING_UNICODE
    /**
     * Construct a string with a WCHAR32 array.
     * @param p The WCHAR32 array.
     */
    NCString( const WCHAR32 *const p );

    /**
     * Construct a string with a WCHAR32 array and a specified length.
     * @param p The WCHAR32 array.
     * @param l Length to set to NCString, (not include the terminator).
     */
    NCString( const WCHAR32 *const p, UINT32 l );
#endif

    /**
     * Construct a string with a CHAR array and specified codepage.
     * @param dwCodepage Code page of the string. See \ref CPIden "Code Page
     Identifiers".
     * @param p The CHAR array.
     * \note When this CHAR array is translated into UTF16, shorter than
     MAX_TRANSLATE_SIZE could perform a fast translate.\n
     Otherwise, it would be slow.
     */
    NCString( UINT32 dwCodePage, const CHAR *const p );

    /**
     * Construct a string with a CHAR array and specified codepage and length.
     * @param dwCodepage Code page of the string. See \ref CPIden "Code Page
     Identifiers".
     * @param p The CHAR array.
     * @param l Length to set to NCString, (not include the terminator).
     * \note When this CHAR array is translated into UTF16, shorter than
     MAX_TRANSLATE_SIZE could perform a fast translate.\n
     Otherwise, it would be slow.
     */
    NCString( UINT32 dwCodepage, const CHAR *const p, UINT32 l );

    /**
     * Distructor
     */
    virtual ~NCString();

    /**
     * Assign a NCString to NCString.
     */
    NCString &operator=( const NCString &rhs );

    /**
     * Concatenate a NCString at the end of this NCString.
     */
    NCString &operator+=( const NCString &rhs );

    /**
     * Set CHAR array into NCString with codepage.
     *
     * @param    p The CHAR array. If NSTRING_UNICODE is not defined, it uses
     * UTF-8 codepage.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL set( const CHAR *const p );

    /**
     * Set CHAR array into NCString with codepage and specified length.
     *
     * @param    p The CHAR array. If NSTRING_UNICODE is not defined, it uses
     * UTF-8 codepage.
     * @param    maxlen The String max length, but this param should be no greater
     * than string length of the "p" param (not include the terminator).
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL set( const CHAR *const p, UINT32 maxlen );

#ifndef NSTRING_UNICODE
    /**
     * Set WCHAR32 array into NCString with codepage.
     *
     * @param    p The WCHAR32 array.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL set( const WCHAR32 *const p );

    /**
     * Set WCHAR32 array into NCString with codepage and specified length.
     *
     * @param    p The WCHAR32 array.
     * @param    l The WCHAR32 array length, (not include the terminator).
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL set( const WCHAR32 *const p, UINT32 l );
#endif

    /**
     * Set CHAR array into NCString with codepage and specified length.
     *
     * @param    dwCodePage The encode of the CHAR array. See \ref CPIden "Code
     Page Identifiers".
     * @param    p The CHAR array.
     * @param    maxlen The String max length, but this param should be no greater
     than string length of the "p" param (not include the terminator).
     * @return   NC_TRUE indicate success, vis versa.
     * \note When this CHAR array is translated into UTF16, shorter than
     MAX_TRANSLATE_SIZE could perform a fast translate.\n
     Otherwise, it would be slow.

     */
    NC_BOOL set( UINT32 dwCodePage, const CHAR *const p, UINT32 maxlen );

    /**
     * \overload
     * Set CHAR array into NCString with codepage.
     *
     * @param    dwCodePage The encode of the CHAR array. See \ref CPIden "Code
     Page Identifiers".
     * @param    p the CHAR array.
     * @return   NC_TRUE indicate success, vis versa.
     * \note When this CHAR array is translated into UTF16, shorter than
     MAX_TRANSLATE_SIZE could perform a fast translate.\n
     Otherwise, it would be slow.
     */
    NC_BOOL set( UINT32 dwCodePage, const CHAR *const p );

    /**
     * Set UTF16 string.
     *
     * @param    p UTF16 string
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16( const UCHAR16 *p );

    /**
     * Set UTF16 coded string. (Endian code according to COMP_OPT_LITTLE_ENDIAN)
     *
     * @param    p UTF16 string
     * @param    l  Length of the string, in WCHAR32 for Windows, or in UINT16 in
     * Linux.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16( const UCHAR16 *p, UINT32 l );

    /**
     * Set UTF16 Big-Endian coded string.
     *
     * @param    p UTF16 Big-Endian coded string
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16BE( const UCHAR16 *p );

    /**
     * Set UTF16 Big-Endian coded string.
     *
     * @param    p UTF16 Big-Endian coded string
     * @param    l  Length of the string, in WCHAR32 for Windows, or in UINT16 in
     * Linux.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16BE( const UCHAR16 *p, UINT32 l );

    /**
     * Set UTF16 Little-Endian coded string.
     *
     * @param    p UTF16 Little-Endian coded string
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16LE( const UCHAR16 *p );

    /**
     * Set UTF16 Little-Endian coded string.
     *
     * @param    p UTF16 Little-Endian coded string
     * @param    l  Length of the string, in WCHAR32 for Windows, or in UINT16 in
     * Linux.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16LE( const UCHAR16 *p, UINT32 l );

    /**
     * Set UTF16 coded string with a BOM symbol ahead.
     *
     * @param    p UTF16 coded string with a BOM symbol ahead
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16BOM( const UCHAR16 *p );

    /**
     * Set UTF16 coded string with a BOM symbol ahead.
     *
     * @param    p UTF16 coded string with a BOM symbol ahead
     * @param    l Length of the string include BOM, in WCHAR32 for Windows, or in
     * UINT16 in Linux.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL setUTF16BOM( const UCHAR16 *p, UINT32 l );

    /**
     * GetString
     *
     * @param    p Buffer head to receive the CHAR array. If NSTRING_UNICODE is
     * not defined, it uses UTF-8 codepage.
     * @param    l Size of the buffer, (include the terminator).
     * @return   The received CHAR array length.
     */
    UINT32 getString( CHAR *p, UINT32 l ) const;

#ifndef NSTRING_UNICODE
    /**
     * GetString
     *
     * @param    p Buffer head to receive the WCHAR32 array.
     * @param    l Size of the buffer.
     * @return   The received CHAR array length.
     */
    UINT32 getString( WCHAR32 *p, UINT32 l ) const;
#endif

    /**
     * GetString with codepage.
     *
     * @param    dwCodePage The codepage you want the CHAR array to be encoded.
     * See \ref CPIden "Code Page Identifiers".
     * @param    p Buffer head to receive the CHAR array.
     * @param    l Size of the buffer, (include the terminator).
     * @return   The received CHAR array length.
     */
    UINT32 getString( UINT32 dwCodePage, CHAR *p, UINT32 l ) const;

    /**
     * GetStringLength with codepage.
     *
     * @param    dwCodePage The codepage you want to be encoded. See \ref CPIden
     * "Code Page Identifiers".
     * @return   the max length to be encoded.
     */
    UINT32 getStringLength( UINT32 dwCodePage ) const;

    /**
     * Get UTF16 string. (Endian code according to COMP_OPT_LITTLE_ENDIAN)
     *
     * @param    p Buffer to put the string.
     * @param    l Buffer size in WCHAR32 for Windows, or in UINT16 in Linux.
     * @return   If the p is NULL or l is 0, returns the buffer size to contain
     this string.\n
     Otherwise, it returns the size written to the buffer. 0 indicate failure.
     */
    UINT32 getUTF16( UCHAR16 *p, UINT32 l ) const;

    /**
     * Get UTF16 Big_Endian coded string.
     *
     * @param    p Buffer to put the string.
     * @param    l Buffer size in WCHAR32 for Windows, or in UINT16 in Linux.
     * @return   If the p is NULL or l is 0, returns the buffer size which is in
     bytes to contain this string.\n
     Otherwise, it returns the size written to the buffer. 0 indicate failure.
     */
    UINT32 getUTF16BE( UCHAR16 *p, UINT32 l ) const;

    /**
     * Get UTF16 Little_Endian coded string.
     *
     * @param    p Buffer to put the string.
     * @param    l Buffer size in WCHAR32 for Windows, or in UINT16 in Linux.
     * @return   If the p is NULL or l is 0, returns the buffer size which is in
     bytes to contain this string.\n
     Otherwise, it returns the size written to the buffer. 0 indicate failure.
     */
    UINT32 getUTF16LE( UCHAR16 *p, UINT32 l ) const;

    /**
     * Get UTF16 coded string with a BOM symbol ahead.
     *
     * @param    p Buffer to put the string.
     * @param    l Buffer size in WCHAR32 for Windows, or in UINT16 in Linux.
     * @return   If the p is NULL or l is 0, returns the buffer size which is in
     bytes to contain this string.\n
     Otherwise, it returns the size written to the buffer. 0 indicate failure.
     */
    UINT32 getUTF16BOM( UCHAR16 *p, UINT32 l ) const;

    /**
     * Add a CHAR array to string.
     *
     * @param    p CHAR array to append. If NSTRING_UNICODE is not defined, it
     * uses UTF-8 codepage.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL add( const CHAR *p );

    /**
     * Add a CHAR array to string.
     *
     * @param    p CHAR array to append. If NSTRING_UNICODE is not defined, it
     * uses UTF-8 codepage.
     * @param    l CHAR length to append, (not include the terminator).
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL add( const CHAR *p, UINT32 l );

#ifndef NSTRING_UNICODE
    /**
     * Add a WCHAR32 array to string.
     *
     * @param    p WCHAR32 array to append.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL add( const WCHAR32 *p );

    /**
     * Add a WCHAR32 array to string.
     *
     * @param    p WCHAR32 array to append.
     * @param    l WCHAR32 length to append, (not include the terminator).
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL add( const WCHAR32 *p, UINT32 l );
#endif

    /**
     * Append one CHAR array to string.
     *
     * @param    dwCodePage Code page of the CHAR array. See \ref CPIden "Code
     * Page Identifiers".
     * @param    p Null-terminated CHAR array.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL add( UINT32 dwCodePage, const CHAR *p );

    /**
     * Add
     *
     * @param    dwCodePage Code page of the CHAR array. See \ref CPIden "Code
     * Page Identifiers".
     * @param    p Char array to append.
     * @param    l Char length to append, (not include the terminator).
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL add( UINT32 dwCodePage, const CHAR *p, UINT32 l );

    // \name Serialize of NCString
    // These several interfaces are used to
    // formate NCString to an organized buffer.
    // Or generate a NCString from an organized NCString buffer.
    // This is used to store the string or transfer it.
    // \par Example
    // \code
    // NCString cString(NTEXT("This is the test string"));
    // UINT32 dwSize = cString.GetSerializeSize();
    // UINT8 *pBuffer = new UINT8[dwSize];
    // cString.Serialize(pBuffer, dwSize) ;
    // NCString cStringRev;
    // cStringRev.Deserialize(pBuffer);
    // We should have cStringRev and cString the same.
    // \endcode
    // \{

    /**
     * Get Serialize Size of this NCString object.
     *
     * @return   Serialized size in bytes.
     */
    virtual UINT32 getSerializeSize() const;

    /**
     * Serialize this object.
     *
     * @param    buffer Buffer to put the serialized object.
     * @param    size The size of the buffer.
     * @return   NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL serialize( UINT8 *const buffer, const UINT32 size ) const;

    /**
     * Deserialize a NCString object from buffer.
     *
     * @param   buffer The buffer which contains a serialized NCString Object.
     * @return   NC_TRUE indicate success, vis versa.
     */
    virtual NC_BOOL deserialize( const UINT8 *const buffer );

    // \}

    /**
     * Get String Buffer
     *
     * @return   CHAR* the CHAR array. If NSTRING_UNICODE is not defined, it uses
     * UTF-8 codepage.
     */
    const CHAR *getString() const;

    /**
     * Get String Buffer length, (not include the terminator).
     *
     * @return   INT32 the CHAR array length. If NSTRING_UNICODE is not defined,
     * it uses UTF-8 codepage.
     */
    UINT32 getLength() const;

    /**
     * This interface is used to implicit convert NCString to a CHAR array.
     */
    operator const CHAR *() const;

    /**
     * Compare two NCString whether equal.
     */
    NC_BOOL operator==( const NCString &lhr ) const;

    /**
     * Compare two NCString whether not equal.
     */
    NC_BOOL operator!=( const NCString &lhr ) const;

    /**
     * Compare two NCString whether this less than lhr.
     */
    NC_BOOL operator<( const NCString &lhr ) const;

    /**
     * Compare two NCString whether this bigger than lhr.
     */
    NC_BOOL operator>( const NCString &lhr ) const;

    /**
     * Set formatted data into NCString.
     *
     * @param    maxlen the formatted result's string len.
     * @param    format format string.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL format( UINT32 maxlen, const CHAR *const formats, ... );

    /**
     * Set formatted data into NCString.
     *
     * @param    format format string.
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL format( const CHAR *const formats, ... );

    /**
     * Convert lower case to uppercase.
     *
     * @notice converting supports all language, so it's poor performance.
     *         if only for ASCII code, please use @tolower.
     *
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL upperCase();

    /**
     * Convert uppercase to lowercase.
     *
     * @notice converting supports all language, so it's poor performance.
     *         if only for ASCII code, please use @toupper.
     *
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL lowerCase();

    /**
     * Convert ASCII code to lowercase.
     *
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL toUpper();

    /**
     * Convert ASCII code to lowercase.
     *
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL toLower();

    /**
     * Convert half width to full width.
     *
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL fullwidth();

    /**
     * Convert full width to half width.
     *
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL halfwidth();

    /**
     * Replace the character with \a ch on the specified position \a pos
     *
     * @param[in]    pos    The position of the character to be replaced. The
     * first is 0.
     * @param[in]    ch    The character to be replaced to.
     *
     * @return   NC_TRUE indicate success, vis versa.
     */
    NC_BOOL replace( UINT32 pos, CHAR ch );

    /**
     * Trim the leading spaces
     *
     */
    NC_BOOL trimLeft();

    /**
     * Trim the trailing spaces
     *
     */
    NC_BOOL trimRight();

    /**
     * Trim the both leading and trailing spaces
     *
     */
    NC_BOOL trim();

    /**
     * Trim the leading spaces
     *
     */
    NC_BOOL trimLeftEx( NC_Judger pfunc_forward );

    /**
     * Trim the trailing spaces
     *
     */
    NC_BOOL trimRightEx( NC_Judger pfunc_backward );

    /**
     * Trim the both leading and trailing spaces
     *
     */
    NC_BOOL trimEx( NC_Judger pfunc_forward, NC_Judger pfunc_backward );

    /**
     * Check whether the string is end with the specified character \a ch
     */
    NC_BOOL endWith( CHAR ch ) const;

    CHAR     operator[]( UINT32 offset ) const;
    NCString left( UINT32 len ) const;
    // NCString right(UINT32 len) const;
    NCString right( UINT32 len ) const;
    NC_BOOL  takeLeft( UINT32 len );
    NC_BOOL  takeRight( UINT32 len );
    NCString mid( INT32 first ) const;
    NCString mid( INT32 first, UINT32 len ) const;
    NC_BOOL  takeMid( INT32 first );
    NC_BOOL  takeMid( INT32 first, UINT32 len );

    INT32 replace( const CHAR *const src, const CHAR *const des );

    INT32 find( const CHAR *const substr ) const;
    INT32 find( const CHAR *const substr, UINT32 start ) const;

    NCString left( NC_Judger pfunc_forward ) const;
    NCString left2( NC_Judger pfunc_backward ) const;
    NCString right( NC_Judger pfunc_backward ) const;

    virtual NCString *clone();
    virtual VOID      copy( NCString *const src );

    // \cond
   protected:
    // Switch between big-endian and little-endian.
    VOID switchEndian( UCHAR16 *const pBuffer, UINT32 nLength ) const;
    // Get UTF16 buffer length.
    UINT32 UTF16strlen( const UCHAR16 *p ) const;
    // Release the string buffer.
    virtual VOID    releaseStrBuff();
    virtual CHAR *  NC_StrAlloc( size_t );
    virtual VOID    NC_StrFree( CHAR *const p );
    virtual CHAR *  NC_StrRealloc( CHAR *str, size_t newlen );
    virtual NC_BOOL isNullBuffer() const;

    struct data {
        data();
        // < CHAR array.If NSTRING_UNICODE is not defined,
        // it uses UTF-8 codepage.
        CHAR *         strBuff;
        UINT32         strLen;    // < equal CHAR array length - 1
        volatile INT32 refCount;  // < reference count
    };

   private:
    struct data *m_data;  // < storing the string data
                          // \endcond
};

// This global operator is used to concatenate two NCString together.
__attribute__( ( visibility( "default" ) ) ) const NCString operator+( const NCString &lhs,
                                                                       const NCString &rhs );

// \name Interfaces for compare NCString and CHAR array
// These interfaces is used to stop compiler
// converting NCString to CHAR* and compare the memory address.
// @{
// Compare NCString to CHAR array.
__attribute__( ( visibility( "default" ) ) ) NC_BOOL operator==( const NCString &  lhs,
                                                                 const CHAR *const rhs );
// Compare CHAR array to NCString.
__attribute__( ( visibility( "default" ) ) ) NC_BOOL operator==( const CHAR *const lhs,
                                                                 const NCString &  rhs );
// Compare NCString to CHAR array for not equal.
__attribute__( ( visibility( "default" ) ) ) NC_BOOL operator!=( const NCString &  lhs,
                                                                 const CHAR *const rhs );
// Compare CHAR array to NCString for not equal.
__attribute__( ( visibility( "default" ) ) ) NC_BOOL operator!=( const CHAR *const lhs,
                                                                 const NCString &  rhs );
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCSTRING_H_
        /* EOF */