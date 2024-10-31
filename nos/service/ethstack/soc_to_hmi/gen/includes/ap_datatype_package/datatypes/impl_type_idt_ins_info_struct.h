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
 * @file impl_type_idt_ins_info_struct.h
 * @brief 
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_INS_INFO_STRUCT_H_
#define AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_INS_INFO_STRUCT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ap_datatype_package/datatypes/impl_type_bool_ref.h"
#include "ap_datatype_package/datatypes/impl_type_double_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_hafheader_struct_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_point3d_struct_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint16_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint32_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint8_ref.h"
#include "ara/com/serializer/transformation_reflection.h"
namespace ap_datatype_package {
namespace datatypes {
struct IDT_Ins_Info_Struct {
    ::ap_datatype_package::datatypes::IDT_HafHeader_Struct_ref header;
    ::ap_datatype_package::datatypes::bool_ref isValid;
    ::ap_datatype_package::datatypes::uint8_ref padding_u8_1;
    ::ap_datatype_package::datatypes::uint16_ref sysStatus;
    ::ap_datatype_package::datatypes::uint16_ref gpsStatus;
    ::ap_datatype_package::datatypes::uint16_ref padding_u16_1;
    ::ap_datatype_package::datatypes::uint32_ref gpsWeek;
    ::ap_datatype_package::datatypes::double_ref gpsSec;
    ::ap_datatype_package::datatypes::double_ref wgsLatitude;
    ::ap_datatype_package::datatypes::double_ref wgslongitude;
    ::ap_datatype_package::datatypes::double_ref wgsAltitude;
    ::ap_datatype_package::datatypes::double_ref wgsheading;
    ::ap_datatype_package::datatypes::double_ref j02Latitude;
    ::ap_datatype_package::datatypes::double_ref j02Longitude;
    ::ap_datatype_package::datatypes::double_ref j02Altitude;
    ::ap_datatype_package::datatypes::double_ref j02heading;
    ::ap_datatype_package::datatypes::IDT_Point3d_Struct_ref sdPosition;
};
} // namespace datatypes
} // namespace ap_datatype_package


STRUCTURE_REFLECTION_DEF(::ap_datatype_package::datatypes::IDT_Ins_Info_Struct,header,isValid,padding_u8_1,sysStatus,gpsStatus,padding_u16_1,gpsWeek,gpsSec,wgsLatitude,wgslongitude,wgsAltitude,wgsheading,j02Latitude,j02Longitude,j02Altitude,j02heading,sdPosition);

#endif // AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_INS_INFO_STRUCT_H_
/* EOF */