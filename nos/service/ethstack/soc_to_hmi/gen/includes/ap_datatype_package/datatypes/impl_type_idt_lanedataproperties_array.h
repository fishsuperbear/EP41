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
 * @file impl_type_idt_lanedataproperties_array.h
 * @brief 
 * @date  
 *
 */
#ifndef AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LANEDATAPROPERTIES_ARRAY_H_
#define AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LANEDATAPROPERTIES_ARRAY_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ap_datatype_package/datatypes/impl_type_idt_lanedata_struct_ref.h"
#include "ara/core/array.h"

namespace ap_datatype_package {
namespace datatypes {
using IDT_LaneDataProperties_Array = ara::core::Array<::ap_datatype_package::datatypes::IDT_LaneData_Struct_ref, 8>;

} // namespace datatypes
} // namespace ap_datatype_package


#endif // AP_DATATYPE_PACKAGE_DATATYPES_IMPL_TYPE_IDT_LANEDATAPROPERTIES_ARRAY_H_
/* EOF */