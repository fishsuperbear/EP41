#ifndef HW_IMPL_PLAT_LOGDEF_H
#define HW_IMPL_PLAT_LOGDEF_H

#include "hw_impl_plat_basic.h"

#define HW_IMPL_PLAT_LOGOUTPUT_INNER_LOGBUF_BYTECOUNT_MAX		256

/*
* Output only when unexpected cases occurs.
* Add to the end of the file.
* The inner log file path can be changed by hw_plat_innerlog_path_change.
* The default inner log file path is HW_PLAT_LOGOUTPUT_INNER_PATH_DEFAULT.
* It will not add any byte like '\r' '\n' at the end. You should add them as you 
* need.
* Because it will only output when unexpected cases, we do not need to care too 
* much about the performance.
* We call fopen fwrite fflush fclose every time we call the function.
* return 0 means has output the log.
* return <0 means cannot output the log mainly due to output off.
* We use stack to put the log buffer, the size is 
* HW_IMPL_PLAT_LOGOUTPUT_INNER_LOGBUF_BYTECOUNT_MAX.
* You cannot call hw_impl_plat_system function in the implementation.
*/
s32 hw_impl_plat_loginner_output(const char* i_pformat, ...);
/*
* The same as hw_impl_plat_loginner_output.
* Only difference is not check outputstage it must output log.
*/
s32 hw_impl_plat_loginner_output_without_checkoutputstage(const char* i_pformat, ...);
/*
* See hw_impl_plat_loginner_output_without_checkoutputstage notes.
*/
s32 hw_impl_plat_loginner_output_valist_without_checkoutputstage(const char* i_pformat, va_list i_valist);

/*
* When we finish output the details we turn off the inner log so that we can 
* find out the earliest error and will not output so much duplicated log.
* It will output "\r\n\r\noutputoff\r\n\r\n" and stop inner output operation.
* You can hw_impl_plat_loginner_outputoff only once after loaded the so.
* return 0 means successfully change the status to output off.
* return <0 means the origin status is already output off.
* You cannot call hw_impl_plat_system function in the implementation.
*/
s32 hw_impl_plat_loginner_outputoff();

/*
* return 0 means WIFEXITED is true and WEXITSTATUS is 0.
* return -1 means the other situation.
* It will not output log when run success.
* Currently, it will not output any log when run fail.
* When input o_pexitstatus is not NULL, *o_pexitstatus is the WEXITSTATUS value 
* of system return.
* When input o_pexitstatus is NULL, it will not output the WEXITSTATUS value.
*/
s32 hw_impl_plat_system(char* i_pcmd, s32* o_pexitstatus);

/*
* Currently, use fopen to open file.
*/
s32 hw_impl_plat_file_open(struct hw_plat_filehandle_t* io_pfilehandle, char* i_filepath);
s32 hw_impl_plat_file_close(struct hw_plat_filehandle_t* io_pfilehandle);
/*
* Output FILE* to *o_ppfile from i_pfilehandle.
*/
s32 hw_impl_plat_file_getfilep(struct hw_plat_filehandle_t* i_pfilehandle, FILE** o_ppfile);
s32 hw_impl_plat_file_setcasaddmode(struct hw_plat_filehandle_t* io_pfilehandle);

#endif
