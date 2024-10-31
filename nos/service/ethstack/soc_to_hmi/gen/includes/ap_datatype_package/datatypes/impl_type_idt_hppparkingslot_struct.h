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
 * @file impl_type_idt_hppparkingslot_struct.h
 * @brief 
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_HPPPARKINGSLOT_STRUCT_H_
#define AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_HPPPARKINGSLOT_STRUCT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ap_datatype_package/datatypes/impl_type_idt_hppobjectdim_struct_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint16_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint8_ref.h"
#include "ara/com/serializer/transformation_reflection.h"
namespace ap_datatype_package {
namespace datatypes {
struct IDT_HPPParkingslot_Struct {
    ::ap_datatype_package::datatypes::uint16_ref ID;
    ::ap_datatype_package::datatypes::IDT_HPPObjectDim_Struct_ref p0;
    ::ap_datatype_package::datatypes::IDT_HPPObjectDim_Struct_ref p1;
    ::ap_datatype_package::datatypes::IDT_HPPObjectDim_Struct_ref p2;
    ::ap_datatype_package::datatypes::IDT_HPPObjectDim_Struct_ref p3;
    ::ap_datatype_package::datatypes::uint8_ref slot_type;
    ::ap_datatype_package::datatypes::uint8_ref slot_occupied;
};
} // namespace datatypes
} // namespace ap_datatype_package


STRUCTURE_REFLECTION_DEF(::ap_datatype_package::datatypes::IDT_HPPParkingslot_Struct,ID,p0,p1,p2,p3,slot_type,slot_occupied);

#endif // AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_HPPPARKINGSLOT_STRUCT_H_
/* EOF */