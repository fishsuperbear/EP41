/**

* Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.

* All Rights Reserved.

*

* Redistribution and use in source and binary forms, with or without

* modification, are NOT permitted except as agreed by

* iAuto Software(Shanghai) Co., Ltd.

*

* Unless required by applicable law or agreed to in writing, software

* distributed under the License is distributed on an "AS IS" BASIS,

* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

*/
#ifndef E2E_INCLUDE_E2EXF_TYPES_H_
#define E2E_INCLUDE_E2EXF_TYPES_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

namespace e2exf {

/// @brief    alias for data identifier
using DataIdentifier = std::uint32_t;

/// @brief    E2E configuration file format
enum class ConfigurationFormat : uint8_t { JSON };

/// @brief    E2E configuration file type
enum class ConfigurationFileType : uint8_t {
    E2E_FILE_TYPE_SM,
    E2E_FILE_TYPE_DATAID,
    E2E_FILE_TYPE_E2EXF
};

}  // namespace e2exf

#endif  // E2E_INCLUDE_E2EXF_TYPES_H_
/* EOF */
