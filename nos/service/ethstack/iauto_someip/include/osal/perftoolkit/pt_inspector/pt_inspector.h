#ifndef INCLUDE_OSAL_PERFTOOLKIT_PT_INSPECTOR_PT_INSPECTOR_H_
#define INCLUDE_OSAL_PERFTOOLKIT_PT_INSPECTOR_PT_INSPECTOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>

#include "pt_string.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*pt_inspector_callback)(pt_string_t* root, const char* opt, void* user_data);

/**
 * @brief register pt_inspector callback
 *
 * @param [in] callback : callback, type: pt_inspector_callback
 * @param [in] category : current category name, must be set(cannot be NULL), max size is 32 bytes
 * @param [in] user_data : user data
 *
 */
void pt_inspector_register_callback(pt_inspector_callback callback, const char* category, void* user_data);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_OSAL_PERFTOOLKIT_PT_INSPECTOR_PT_INSPECTOR_H_
