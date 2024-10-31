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
 * @file impl_type_algcanfdmsgframe.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGCANFDMSGFRAME_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGCANFDMSGFRAME_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_canfd_msg190.h"
#include "hozon/netaos/impl_type_canfd_msg191.h"
#include "hozon/netaos/impl_type_canfd_msg192.h"
#include "hozon/netaos/impl_type_canfd_msg194.h"
#include "hozon/netaos/impl_type_canfd_msg196.h"
#include "hozon/netaos/impl_type_canfd_msg210.h"
#include "hozon/netaos/impl_type_canfd_msg255.h"
#include "hozon/netaos/impl_type_canfd_msg265.h"
#include "hozon/netaos/impl_type_canfd_msg8f.h"
#include "hozon/netaos/impl_type_canfd_msgfe.h"
namespace hozon {
namespace netaos {
struct AlgCanFdMsgFrame {
    ::hozon::netaos::CANFD_Msg8F CANFD_Msg8F;
    ::hozon::netaos::CANFD_MsgFE CANFD_MsgFE;
    ::hozon::netaos::CANFD_Msg190 CANFD_Msg190;
    ::hozon::netaos::CANFD_Msg191 CANFD_Msg191;
    ::hozon::netaos::CANFD_Msg192 CANFD_Msg192;
    ::hozon::netaos::CANFD_Msg194 CANFD_Msg194;
    ::hozon::netaos::CANFD_Msg196 CANFD_Msg196;
    ::hozon::netaos::CANFD_Msg210 CANFD_Msg210;
    ::hozon::netaos::CANFD_Msg265 CANFD_Msg265;
    ::hozon::netaos::CANFD_Msg255 CANFD_Msg255;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgCanFdMsgFrame,CANFD_Msg8F,CANFD_MsgFE,CANFD_Msg190,CANFD_Msg191,CANFD_Msg192,CANFD_Msg194,CANFD_Msg196,CANFD_Msg210,CANFD_Msg265,CANFD_Msg255);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGCANFDMSGFRAME_H_
/* EOF */