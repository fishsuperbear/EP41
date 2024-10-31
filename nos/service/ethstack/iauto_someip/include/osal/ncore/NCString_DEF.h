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
 * @file NCString_DEF.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCSTRING_DEF_H_
#define INCLUDE_NCORE_NCSTRING_DEF_H_

#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE

// Default Charset
// \deprecated This is planned to abandon.
const INT32 NCSTRING_CHARSET_DEFAULT = 0;

// The max size you could store c string in a NCString
// with codepage other than default.
const UINT32 NC_MAX_TRANSLATE_SIZE = MAX_PATH;

// \name Code Page Identifiers
// \anchor CPIden
// \{
// code page is following the windows-codepage
// convert name is not the same,
// but they are using the same converter

// Chinese Code page GB18030. Compatible with GB2312 and GBK.
const UINT32 NC_CP_GB18030 = 54936U;

// Chinese Code page GBK. Compatible with GB2312.
const UINT32 NC_CP_GBK = 936U;

// English Code page ANSI - Latin I for English.
const UINT32 NC_CP_LANTI = 1252U;

// Japanese Code page Shift-JIS.
const UINT32 NC_CP_SJIS = 932U;

// Japanese Code page MS JIS KANJI
const UINT32 NC_CP_MSJIS = NC_CP_SJIS;

// Unicode Code page UTF-8
const UINT32 NC_CP_UTF8 = 65001U;

// 7-bit ASCII Code page
const UINT32 NC_CP_US_ASCII_7BIT = 20127U;

// English Code page ISO-Latin I, ISO 8859-1
const UINT32 NC_CP_ISO_LANTI = 28591U;

// Central Europe, ISO 8859-2
const UINT32 NC_CP_ISO_EUROPE = 28592U;

// English Code page ISO-Latin 3, ISO 8859-3
const UINT32 NC_CP_ISO_LANT3 = 28593U;

// ISO 8859-4 Baltic
const UINT32 NC_CP_ISO_BALTIC = 28594U;

// ISO 8859-5 Cyrillic
const UINT32 NC_CP_ISO_CYRILLIC = 28595U;

// ISO 8859-6 Arabic
const UINT32 NC_CP_ISO_ARABIC = 28596U;

// ISO 8859-7 Greek
const UINT32 NC_CP_ISO_GREEK = 28597U;

// ISO 8859-8 Hebrew
const UINT32 NC_CP_ISO_HEBREW = 28598U;

// ISO 8859-9 Latin 5
const UINT32 NC_CP_ISO_LANT5 = 28599U;

// ISO 8859-15 Latin 9
const UINT32 NC_CP_ISO_LANT9 = 28605U;

// EUC for JP
const UINT32 NC_CP_EUC_JP = 20932U;

// EUC for KR
const UINT32 NC_CP_EUC_KR = 51949U;

// UTF16LE
const UINT32 NC_CP_UCS2_LE = 1200U;

// UTF16BE
const UINT32 NC_CP_UCS2_BE = 1201U;

// windows-874-2000
const UINT32 NC_CP_EUR_THAI = 874U;

// Windows Arabic
const UINT32 NC_CP_EUR_ARABIC = 1256U;

// Windows Vietnamese
const UINT32 NC_CP_EUR_VIETNAMESE = 1258U;

// Mark for UTF16
const UINT32 NC_CP_UTF16 = static_cast<UINT32>( -1 );

// \}
OSAL_END_NAMESPACE

#endif  // INCLUDE_NCORE_NCSTRING_DEF_H_
/* EOF */
