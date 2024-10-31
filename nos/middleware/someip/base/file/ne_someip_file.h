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
#ifndef BASE_FILE_NE_SOMEIP_FILE_H
#define BASE_FILE_NE_SOMEIP_FILE_H

#ifdef  __cplusplus
extern "C" {
#endif

#include <stdbool.h>

//  Definition of file's permission
typedef enum {
    NE_SOMEIP_FILE_PERMISSION_UNKNOWN      = 0x0000,
    NE_SOMEIP_FILE_PERMISSION_OWNERREAD    = 0x0400,    //  the owner can read the file
    NE_SOMEIP_FILE_PERMISSION_OWNERWRITE   = 0x0200,    //  the owner can write the file
    NE_SOMEIP_FILE_PERMISSION_OWNEREXE     = 0x0100,    //  the owner can execute the file
    NE_SOMEIP_FILE_PERMISSION_GROUPREAD    = 0x0040,    //  the member in owner's group can read the file
    NE_SOMEIP_FILE_PERMISSION_GROUPWRITE   = 0x0020,    //  the member in owner's group can write the file
    NE_SOMEIP_FILE_PERMISSION_GROUPEXE     = 0x0010,    //  the member in owner's group can execute the file
    NE_SOMEIP_FILE_PERMISSION_OTHERREAD    = 0x0004,    //  other user can read the file
    NE_SOMEIP_FILE_PERMISSION_OTHERWRITE   = 0x0002,    //  other user can write the file
    NE_SOMEIP_FILE_PERMISSION_OTHEREXE     = 0x0001,    //  other user can execute the file
    NE_SOMEIP_FILE_PERMISSION_MASK         = 0xFFFF,    //  permission mask
} ne_someip_file_permission_t;

bool ne_someip_file_is_exist(const char* file);
bool ne_someip_file_create_dir(const char* file);
bool ne_someip_file_set_permission(const char* file, unsigned int dwPerms);

#ifdef  __cplusplus
}
#endif
#endif  // BASE_FILE_NE_SOMEIP_FILE_H
/* EOF */
