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
 * @file impl_type_idt_adas_dataproperties_struct.h
 * @brief 
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_ADAS_DATAPROPERTIES_STRUCT_H_
#define AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_ADAS_DATAPROPERTIES_STRUCT_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ap_datatype_package/datatypes/impl_type_idt_dynamicsrobject_array_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_functionstate_struct_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_imudata_struct_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_lanedataproperties_array_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_locfusioninfo_struct_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_poscoordlocal_array_ref.h"
#include "ap_datatype_package/datatypes/impl_type_idt_staticsrobject_array_ref.h"
#include "ap_datatype_package/datatypes/impl_type_uint64_ref.h"
#include "ara/com/serializer/transformation_reflection.h"
namespace ap_datatype_package {
namespace datatypes {
struct IDT_ADAS_Dataproperties_Struct {
    ::ap_datatype_package::datatypes::IDT_LocFusionInfo_Struct_ref locFusionInfo;
    ::ap_datatype_package::datatypes::IDT_PosCoordLocal_Array_ref decisionInfo;
    ::ap_datatype_package::datatypes::IDT_FunctionState_Struct_ref functionstate;
    ::ap_datatype_package::datatypes::IDT_DynamicSRObject_Array_ref dynamicSRData;
    ::ap_datatype_package::datatypes::IDT_StaticSRObject_Array_ref staticSRData;
    ::ap_datatype_package::datatypes::IDT_LaneDataProperties_Array_ref laneData;
    ::ap_datatype_package::datatypes::IDT_IMUdata_Struct_ref imudata;
    ::ap_datatype_package::datatypes::uint64_ref timestamp;
};
} // namespace datatypes
} // namespace ap_datatype_package


STRUCTURE_REFLECTION_DEF(::ap_datatype_package::datatypes::IDT_ADAS_Dataproperties_Struct,locFusionInfo,decisionInfo,functionstate,dynamicSRData,staticSRData,laneData,imudata,timestamp);

#endif // AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_ADAS_DATAPROPERTIES_STRUCT_H_
/* EOF */