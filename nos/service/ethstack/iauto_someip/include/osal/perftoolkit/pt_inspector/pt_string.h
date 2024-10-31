#ifndef INCLUDE_OSAL_PERFTOOLKIT_PT_INSPECTOR_PT_STRING_H_
#define INCLUDE_OSAL_PERFTOOLKIT_PT_INSPECTOR_PT_STRING_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>

#ifdef __cplusplus

typedef struct pt_string pt_string_t;

#include <string>
void pt_string_append_string(pt_string_t* node, const std::string& str);

extern "C" {
#endif

typedef struct pt_string pt_string_t;

// default tab is 2 spaces
void pt_string_push_tab();
void pt_string_pop_tab();

// the node will be passed via callback: pt_inspector_callback
void pt_string_append_format(pt_string_t* node, const char* format, ...);
void pt_string_append_cstr(pt_string_t* node, const char* str);
void pt_string_append_vformat(pt_string_t* node, const char* format, va_list args);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_OSAL_PERFTOOLKIT_PT_INSPECTOR_PT_STRING_H_
