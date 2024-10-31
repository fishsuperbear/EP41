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
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "ne_someip_log.h"
#include "ne_someip_file.h"

bool ne_someip_file_is_exist(const char* file)
{
    int ret = access(file, F_OK);
    if (0 == ret) {
        ne_someip_log_info("[%s] is exist", file);
        return true;
    } else {
        ne_someip_log_info("[%s] is not exist", file);
        return false;
    }
}

bool ne_someip_file_create_dir(const char* file)
{
    int ret = mkdir(file, 0755);
    if (0 == ret) {
        ne_someip_log_info("create [%s] success", file);
        return true;
    } else {
        ne_someip_log_info("create [%s] failed", file);
        return false;
    }
}

bool ne_someip_file_set_permission(const char* file, unsigned int dwPerms)
{
    mode_t mode = 0;
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_OWNERREAD) {
        mode |= S_IRUSR;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_OWNERWRITE) {
        mode |= S_IWUSR;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_OWNEREXE) {
        mode |= S_IXUSR;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_GROUPREAD) {
        mode |= S_IRGRP;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_GROUPWRITE) {
        mode |= S_IWGRP;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_GROUPEXE) {
        mode |= S_IXGRP;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_OTHERREAD) {
        mode |= S_IROTH;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_OTHERWRITE) {
        mode |= S_IWOTH;
    }
    if (dwPerms & NE_SOMEIP_FILE_PERMISSION_OTHEREXE) {
        mode |= S_IXOTH;
    }

    int ret = chmod(file, mode);
    if (0 == ret) {
        return true;
    } else {
        ne_someip_log_error("chmod [%s] error", file);
        return false;
    }
}
/* EOF */