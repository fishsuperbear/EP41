#ifndef INCLUDE_OSAL_PERFTOOLKIT_PT_STAT_H_
#define INCLUDE_OSAL_PERFTOOLKIT_PT_STAT_H_

#ifdef __cplusplus
extern "C" {
#endif

void* pt_stat_new(const char* name);
void pt_stat_free(void* handle);
void pt_stat_commit(void* handle, int64_t size);
// void pt_stat_query(void* handle);

#ifdef __cplusplus
}
#endif

#endif  // INCLUDE_OSAL_PERFTOOLKIT_PT_STAT_H_

