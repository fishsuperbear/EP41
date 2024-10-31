#include "hw_impl_plat_logdump.h"

s32 hw_impl_plat_logdump_backtrace()
{
	u32 onstatus;
	hw_plat_logoutput_getonoffstatus(&onstatus);
	if (onstatus == 0)
	{
		return -1;
	}
	return hw_impl_plat_logdump_backtrace_without_checkoutputstage();
}

s32 hw_impl_plat_logdump_backtrace_without_checkoutputstage()
{
	void* array[100];
	s32 size;
	char** strings;
	s32 i;

	size = backtrace(array, 100);
	strings = (char**)backtrace_symbols(array, size);

	for (i = 0; i < size; i++) {
		hw_impl_plat_loginner_output_without_checkoutputstage("%d %s\r\n", i, strings[i]);
	}
	free(strings);
	return 0;
}
