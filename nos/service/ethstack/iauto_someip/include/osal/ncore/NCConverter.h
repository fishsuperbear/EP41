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
 * @file NCConverter.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCCONVERTER_H_
#define INCLUDE_NCORE_NCCONVERTER_H_

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
// Default Code page.
const UINT32 NC_CP_ACP = 0U;  // ANSII
// Mark for UTF16, UTF32
const UINT32 NC_CP_UTF16LE = static_cast<UINT32>( -2 );
const UINT32 NC_CP_UTF16BE = static_cast<UINT32>( -3 );
const UINT32 NC_CP_UTF32BE = static_cast<UINT32>( -4 );
const UINT32 NC_CP_UTF32LE = static_cast<UINT32>( -5 );
// Max CodePage Number (supported)
const UINT32 NC_CP_MAX = 23U;

/**
 * @brief
 *
 * @class NCConverter
 */
class __attribute__( ( visibility( "default" ) ) ) NCConverter {
   public:
    /**
     * @brief change UTF8 to char
     *
     * @param dwCodePage encoding format
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 UTF8ToChar( const UINT32 &dwCodePage, const CHAR *pszSrc, const UINT32 &nSrcLen,
                              CHAR *pszDst, const UINT32 &nDstLen );

    /**
     * @brief change UTF8 to wchar
     *
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 UTF8ToWChar( const CHAR *pszSrc, const UINT32 &nSrcLen, WCHAR32 *pszDst,
                               const UINT32 &nDstLen );

    /**
     * @brief change char to UTF8
     *
     * @param dwCodePage encoding format
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 CharToUTF8( const UINT32 &dwCodePage, const CHAR *pszSrc, const UINT32 &nSrcLen,
                              CHAR *pszDst, const UINT32 &nDstLen );

    /**
     * @brief change wchar to UTF8
     *
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 WCharToUTF8( const WCHAR32 *pszSrc, const UINT32 &nSrcLen, CHAR *pszDst,
                               const UINT32 &nDstLen );

    /**
     * @brief change UTF16BE to UTF8
     *
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 UTF16BEToUTF8( const UCHAR16 *pszSrc, const UINT32 &nSrcLen, CHAR *pszDst,
                                 const UINT32 &nDstLen );

    /**
     * @brief change UTF16LE to UTF8
     *
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 UTF16LEToUTF8( const UCHAR16 *pszSrc, const UINT32 &nSrcLen, CHAR *pszDst,
                                 const UINT32 &nDstLen );

    /**
     * @brief change UTF8 to UTF16BE
     *
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 UTF8ToUTF16BE( const CHAR *pszSrc, const UINT32 &nSrcLen, UCHAR16 *pszDst,
                                 const UINT32 &nDstLen );

    /**
     * @brief change UTF8 to UTF16LE
     *
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 UTF8ToUTF16LE( const CHAR *pszSrc, const UINT32 &nSrcLen, UCHAR16 *pszDst,
                                 const UINT32 &nDstLen );

    /**
     * @brief change UTF16 to char
     *
     * @param dwCodePage encoding format
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 UTF16ToChar( const UINT32 &dwCodePage, const UCHAR16 *pszSrc,
                               const UINT32 &nSrcLen, CHAR *pszDst, const UINT32 &nDstLen );

    /**
     * @brief change char to UTF16
     *
     * @param dwCodePage encoding format
     * @param pszSrc pointr of source data
     * @param nSrcLen length of source data
     * @param pszDst buffer for save new data
     * @param nDstLen length for pszDst
     * @return UINT32 the length of the converted data
     */
    static UINT32 CharToUTF16( const UINT32 &dwCodePage, const CHAR *pszSrc, const UINT32 &nSrcLen,
                               UCHAR16 *pszDst, const UINT32 &nDstLen );

    /**
     * @brief change string to upper case
     *
     * @param src pointr of source data
     * @param srcLen length of source data
     * @param dst buffer for save new data
     * @param dstCapacity length for dst
     * @return SCHAR returned value
     */
    static SCHAR StrToUpper( const CHAR *src, UINT32 srcLen, CHAR *dst, UINT32 dstCapacity );

    /**
     * @brief change string to lower case
     *
     * @param src pointr of source data
     * @param srcLen length of source data
     * @param dst buffer for save new data
     * @param dstCapacity length for dst
     * @return SCHAR returned value
     */
    static SCHAR StrToLower( const CHAR *src, UINT32 srcLen, CHAR *dst, UINT32 dstCapacity );

    /**
     * @brief change half angle to full angle
     *
     * @param src pointr of source data
     * @param srcLen length of source data
     * @param dst buffer for save new data
     * @param dstCapacity length for dst
     * @return SCHAR returned value
     */
    static SCHAR HalfToFullwidth( const CHAR *src, UINT32 srcLen, CHAR *dst, UINT32 dstCapacity );

    /**
     * @brief change full angle to half angle
     *
     * @param src pointr of source data
     * @param srcLen length of source data
     * @param dst buffer for save new data
     * @param dstCapacity length for dst
     * @return SCHAR returned value
     */
    static SCHAR FullToHalfwidth( const CHAR *src, UINT32 srcLen, CHAR *dst, UINT32 dstCapacity );

   private:
    struct CONV_CP_INFO {
        const UINT32 dwToCP;
        const CHAR * pszToCP;
        const UINT32 dwFromCP;
        const CHAR * pszFromCP;
    };

    /**
     * @brief
     *
     * @param cp_info
     * @param pszDst
     * @param nDstLen
     * @param pszSrc
     * @param nSrcLen
     * @return UINT32
     */
    static UINT32 convert( const CONV_CP_INFO &cp_info, CHAR *pszDst, const UINT32 &nDstLen,
                           const CHAR *pszSrc, const UINT32 &nSrcLen );

    /**
     * @brief get the cpInfo name
     *
     * @param dwCodePage encoding format
     * @param szCPName buffer to save cp name
     * @param nStrLen length of szCPName
     * @return VOID
     */
    static VOID getCPName( const UINT32 dwCodePage, CHAR *const szCPName, UINT32 nStrLen );

   private:
    NCConverter();
    virtual ~NCConverter();

    NCConverter( const NCConverter & );
    NCConverter &operator=( const NCConverter & );
};

OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCCONVERTER_H_
/* EOF */
