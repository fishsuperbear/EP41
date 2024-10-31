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
 * @file impl_type_dtcloud_haffltmgreventbusoutarray.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFFLTMGREVENTBUSOUTARRAY_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFFLTMGREVENTBUSOUTARRAY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include "hozon/netaos/impl_type_dtcloud_hafcomdatafltinfo.h"
#include "hozon/netaos/impl_type_dtcloud_hafpartnrecusysfltinfo.h"
#include "hozon/netaos/impl_type_dtcloud_hafplathealthmngrinfo.h"
#include "hozon/netaos/impl_type_dtcloud_hafsocclustfltvalinfo.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_HafFltMgrEventBusOutArray {
    std::uint8_t isFltMgrValidSt;
    ::hozon::netaos::DtCloud_HafsocClustFltValInfo socClustFltValInfo;
    ::hozon::netaos::DtCloud_HafcomDataFltInfo comDataFltInfo;
    ::hozon::netaos::DtCloud_HafpartnrEcuSysFltInfo partnrEcuSysFltInfo;
    ::hozon::netaos::DtCloud_HafplatHealthMngrInfo platHealthMngrInfo;
    std::uint8_t mcuPlatFltValInfo;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_HafFltMgrEventBusOutArray,isFltMgrValidSt,socClustFltValInfo,comDataFltInfo,partnrEcuSysFltInfo,platHealthMngrInfo,mcuPlatFltValInfo);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFFLTMGREVENTBUSOUTARRAY_H_
/* EOF */