/**
 * Copyright @ 2019 iAuto (Shanghai) Co., Ltd.
 * All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are NOT permitted except as agreed by
 * iAuto (Shanghai) Co., Ltd.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <unistd.h>
#include <stdio.h>
#include "ne_someip_daemon.h"

int main(int argc, char* argv[])
{
    ne_someip_daemon_t* daemon = ne_someip_daemon_init();
	if (NULL == daemon) {
		DEBUG_LOG("someip daemon init error");
		return 0;
	}

    sleep(3000000000);

    return 0;
}
/* EOF */
