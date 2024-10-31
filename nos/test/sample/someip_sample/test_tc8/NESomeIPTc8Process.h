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
/**
 * @file NESomeIPTc8Process.h
 * @brief Declaration file of class NESomeIPTc8Process.
 */

#ifndef TC8TEST_NESOMEIPTC8PROCESS_H_
#define TC8TEST_NESOMEIPTC8PROCESS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include "ne_someip_server_context.h"
#include "ne_someip_config_parse.h"

bool init();
void stopSeviceIns2000();
void startSeviceIns2000(char* config_path);
int OfferService(uint16_t service, uint16_t NumInstance);
int StopService(uint16_t service);
int TriggerEvent(uint16_t service, uint16_t EventGroup, uint16_t EventId);

#ifdef __cplusplus
}
#endif
#endif  // TC8TEST_NESOMEIPTC8PROCESS_H_
/* EOF */
