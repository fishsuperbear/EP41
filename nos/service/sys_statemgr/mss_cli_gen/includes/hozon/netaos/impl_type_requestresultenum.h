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
 * @file impl_type_requestresultenum.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_REQUESTRESULTENUM_H_
#define HOZON_NETAOS_IMPL_TYPE_REQUESTRESULTENUM_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include <cstdint>
namespace hozon {
namespace netaos {
enum class RequestResultEnum : std::uint8_t
{
Succ =0,
Fail1 =1,
Fail2 =2,
Fail3 =3
};
} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_IMPL_TYPE_REQUESTRESULTENUM_H_
/* EOF */