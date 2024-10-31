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
 * @file NCGlobalAPI.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCGLOBALAPI_H
#define NCGLOBALAPI_H

/*=============================================================================
 */
#if defined( _WIN32 ) || defined( _WIN32_WCE )
#ifdef NCCORE_EXPORTS
#define NCCORE_API __declspec( dllexport )
#define NCCORE_DATA __declspec( dllexport )
#else
#ifdef NCCORE_IS_SINGLEPROC
#define NCCORE_API
#define NCCORE_DATA
#else
#define NCCORE_API __declspec( dllimport )
#define NCCORE_DATA __declspec( dllimport )
#endif
#endif
#elif defined( linux ) || defined( __linux__ )  // linux
#define NCCORE_API __attribute__( ( visibility( "default" ) ) )
#define NCCORE_DATA __attribute__( ( visibility( "default" ) ) )
#define NUTSHELL_API __attribute__( ( visibility( "default" ) ) )
#define NUTSHELL_DATA __attribute__( ( visibility( "default" ) ) )
#else
#define NCCORE_API
#define NCCORE_DATA

#define NUTSHELL_API
#define NUTSHELL_DATA
#endif

#endif /* NCGLOBALAPI_H */
/* EOF */
