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
 * @file NCNameSpace.h
 * @brief
 * @date 2020-11-30
 *
 */
#ifndef NCNAMESPACE_H
#define NCNAMESPACE_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#ifndef USING_NAMESPACE_OSAL
#define USING_NAMESPACE_OSAL 1
#endif

#if USING_NAMESPACE_OSAL
#define OSAL iautosar::osal

#define OSAL_BEGIN_NAMESPACE \
    namespace iautosar {     \
    namespace osal {

#define OSAL_END_NAMESPACE \
    }                      \
    }

#define OSAL_USING_NAMESPACE using namespace iautosar::osal;
#else
#define OSAL nutshell

#define OSAL_BEGIN_NAMESPACE namespace nutshell {
#define OSAL_END_NAMESPACE }
#define OSAL_USING_NAMESPACE using namespace nutshell;
#endif

#endif /* NCNAMESPACE_H */
