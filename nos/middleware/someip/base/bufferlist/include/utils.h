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
#ifndef __AP_UTILS_H__
#define __AP_UTILS_H__




//#define LOG_ENABLE 1
#ifdef LOG_ENABLE
    #define MLOGD(para, args...) printf(para, ##args)
    #define MLOGE(para, args...) printf(para, ##args)
#else
    #define MLOGD(para, args...)
    #define MLOGE(para, args...) printf(para, ##args)
#endif



#define MEM_CHECK_POINTER_RET_VOID(POINTER) \
    if (NULL == POINTER) { \
        MLOGE("\033[1;40;31mUT[%s][%d]: check pointer error. \n\033[0m", __FUNCTION__, __LINE__); \
        return; \
    }
#define MEM_CHECK_POINTER_RET_PTR(POINTER) \
    if (NULL == POINTER) { \
       MLOGE("\033[1;40;31mUT[%s][%d]: check pointer error. \n\033[0m", __FUNCTION__, __LINE__); \
        return NULL; \
    }
#define MEM_CHECK_POINTER_RET_INTEGER(POINTER) \
    if (NULL == POINTER) { \
        MLOGE("\033[1;40;31mUT[%s][%d]: check pointer error. \n\033[0m", __FUNCTION__, __LINE__); \
        return 1; \
    }
#define MEM_CHECK_LIST_EMPETY_RET_VOID(LIST) \
    if (memory_list_empty(LIST)) { \
        MLOGE("\033[1;40;31mUT[%s][%d]: check list error. \n\033[0m", __FUNCTION__, __LINE__); \
        return; \
    }
#define MEM_CHECK_LIST_EMPETY_RET_INTEGER(LIST) \
    if (memory_list_empty(LIST)) { \
        MLOGE("\033[1;40;31mUT[%s][%d]: check list error. \n\033[0m", __FUNCTION__, __LINE__); \
        return 1; \
    }
#define MEM_CHECK_LIST_EMPETY_RET_PTR(LIST) \
    if (memory_list_empty(LIST)) { \
        MLOGE("\033[1;40;31mUT[%s][%d]: check list error. \n\033[0m", __FUNCTION__, __LINE__); \
        return NULL; \
    }
#define MEM_CHECK_SIZE_LENGTH_RET_PTR(SIZE) \
    if (0 == SIZE) { \
        /* MLOGE("\033[1;40;31mUT[%s][%d]: check size error. \n\033[0m", __FUNCTION__, __LINE__); */\
        return NULL; \
    }

#define MEM_CHECK_SIZE_LENGTH_RET_INTEGER(SIZE) \
    if (0 == SIZE) { \
        MLOGE("\033[1;40;31mUT[%s][%d]: check size error. \n\033[0m", __FUNCTION__, __LINE__); \
        return 1; \
    }

#endif // __AP_UTILS_H__
