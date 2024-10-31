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
 * @file impl_type_idt_locallfusionpos_struct.h
 * @brief 
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LOCALLFUSIONPOS_STRUCT_H_
#define AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LOCALLFUSIONPOS_STRUCT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ap_datatype_package/datatypes/impl_type_float_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint64_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint8_ref.h"
#include "ara/com/serializer/transformation_reflection.h"
namespace ap_datatype_package {
namespace datatypes {
struct IDT_LocAllFusionPos_Struct {
    ::ap_datatype_package::datatypes::uint64_ref ticktime;
    ::ap_datatype_package::datatypes::uint8_ref status;
    ::ap_datatype_package::datatypes::uint8_ref ns;
    ::ap_datatype_package::datatypes::uint8_ref ew;
    ::ap_datatype_package::datatypes::uint8_ref fusiontype;
    ::ap_datatype_package::datatypes::float_ref posEnu_Longitude;
    ::ap_datatype_package::datatypes::float_ref posEnu_latitude;
    ::ap_datatype_package::datatypes::float_ref speed;
    ::ap_datatype_package::datatypes::float_ref course;
    ::ap_datatype_package::datatypes::float_ref alt;
    ::ap_datatype_package::datatypes::float_ref posAcc;
    ::ap_datatype_package::datatypes::float_ref courseAcc;
    ::ap_datatype_package::datatypes::float_ref altAcc;
    ::ap_datatype_package::datatypes::float_ref speedAcc;
    ::ap_datatype_package::datatypes::uint64_ref datetime;
};
} // namespace datatypes
} // namespace ap_datatype_package


STRUCTURE_REFLECTION_DEF(::ap_datatype_package::datatypes::IDT_LocAllFusionPos_Struct,ticktime,status,ns,ew,fusiontype,posEnu_Longitude,posEnu_latitude,speed,course,alt,posAcc,courseAcc,altAcc,speedAcc,datetime);

#endif // AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LOCALLFUSIONPOS_STRUCT_H_
/* EOF */