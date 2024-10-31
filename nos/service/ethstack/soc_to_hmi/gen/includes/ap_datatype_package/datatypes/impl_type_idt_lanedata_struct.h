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
 * @file impl_type_idt_lanedata_struct.h
 * @brief 
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LANEDATA_STRUCT_H_
#define AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LANEDATA_STRUCT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ap_datatype_package/datatypes/impl_type_float_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint8_ref.h"
#include "ara/com/serializer/transformation_reflection.h"
namespace ap_datatype_package {
namespace datatypes {
struct IDT_LaneData_Struct {
    ::ap_datatype_package::datatypes::uint8_ref lane_state;
    ::ap_datatype_package::datatypes::uint8_ref lane_color;
    ::ap_datatype_package::datatypes::uint8_ref lane_type;
    ::ap_datatype_package::datatypes::uint8_ref lane_ID;
    ::ap_datatype_package::datatypes::float_ref lane_equation_C0;
    ::ap_datatype_package::datatypes::float_ref lane_equation_C1;
    ::ap_datatype_package::datatypes::float_ref lane_equation_C2;
    ::ap_datatype_package::datatypes::float_ref lane_equation_C3;
    ::ap_datatype_package::datatypes::float_ref laneWidth;
    ::ap_datatype_package::datatypes::float_ref laneLineWidth;
    ::ap_datatype_package::datatypes::float_ref lane_start_X;
    ::ap_datatype_package::datatypes::float_ref lane_start_Y;
    ::ap_datatype_package::datatypes::float_ref lane_end_X;
    ::ap_datatype_package::datatypes::float_ref lane_end_Y;
};
} // namespace datatypes
} // namespace ap_datatype_package


STRUCTURE_REFLECTION_DEF(::ap_datatype_package::datatypes::IDT_LaneData_Struct,lane_state,lane_color,lane_type,lane_ID,lane_equation_C0,lane_equation_C1,lane_equation_C2,lane_equation_C3,laneWidth,laneLineWidth,lane_start_X,lane_start_Y,lane_end_X,lane_end_Y);

#endif // AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LANEDATA_STRUCT_H_
/* EOF */