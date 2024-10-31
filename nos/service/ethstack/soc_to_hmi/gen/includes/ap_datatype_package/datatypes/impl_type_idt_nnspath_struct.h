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
 * @file impl_type_idt_nnspath_struct.h
 * @brief 
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_NNSPATH_STRUCT_H_
#define AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_NNSPATH_STRUCT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ap_datatype_package/datatypes/impl_type_float_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint16_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint8_ref.h"
#include "ara/com/serializer/transformation_reflection.h"
namespace ap_datatype_package {
namespace datatypes {
struct IDT_NNSPath_Struct {
    ::ap_datatype_package::datatypes::float_ref nns_Lon;
    ::ap_datatype_package::datatypes::float_ref nns_Lat;
    ::ap_datatype_package::datatypes::float_ref nns_High;
    ::ap_datatype_package::datatypes::float_ref nns_Heading;
    ::ap_datatype_package::datatypes::uint8_ref ns;
    ::ap_datatype_package::datatypes::uint8_ref ew;
    ::ap_datatype_package::datatypes::uint16_ref padding_u16_1;
};
} // namespace datatypes
} // namespace ap_datatype_package


STRUCTURE_REFLECTION_DEF(::ap_datatype_package::datatypes::IDT_NNSPath_Struct,nns_Lon,nns_Lat,nns_High,nns_Heading,ns,ew,padding_u16_1);

#endif // AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_NNSPATH_STRUCT_H_
/* EOF */