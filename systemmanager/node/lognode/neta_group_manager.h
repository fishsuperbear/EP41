#ifndef NETA_GROUP_MANAGER_H
#define NETA_GROUP_MANAGER_H

void thread_add_blockgroup(u32 i_groupindex);

// do not use the input i_param currently
s32 clear_group(void* i_param);

void thread_clear_group(void);

#endif