/*                  - Mellanox Confidential and Proprietary -
 *
 *  Copyright (C) Jan 2013, Mellanox Technologies Ltd.  ALL RIGHTS RESERVED.
 *
 *  Except as specifically permitted herein, no portion of the information,
 *  including but not limited to object code and source code, may be reproduced,
 *  modified, distributed, republished or otherwise exploited in any form or by
 *  any means for any purpose without the prior written permission of Mellanox
 *  Technologies Ltd. Use of software subject to the terms and conditions
 *  detailed in the file "LICENSE.txt".
 *
 */

/*
 *
 *  mtcr.h - Mellanox Software tools (mst) driver definitions
 *
 */

#ifndef _MST_H
#define _MST_H

#include "mtcr_com_defs.h"
#include "mtcr_mf.h"


#ifdef __cplusplus
extern "C" {
#endif

#define SLV_ADDRS_NUM 128

#ifdef __WIN__
#define FromHandle(h)   ((MT_ulong_ptr_t)(h))
#define ToHandle(h)     ((HANDLE)(h))
#else
#define FromHandle(h)   ((int)(h))
#define ToHandle(h)     ((HANDLE)(h))
#endif

int set_i2c_freq(mfile* mf, u_int8_t freq);
int get_i2c_freq(mfile* mf, u_int8_t* freq);

/*
 * Get list of MST (Mellanox Software Tools) devices.
 * Put all device names as null-terminated strings to buf.
 *
 * Return number of devices found or -1 if buf overflow
 */
MTCR_API int mdevices(char *buf, int len, int mask);

/*
 * Get list of MST (Mellanox Software Tools) devices info records.
 * Return a dynamic allocated array of dev_info records.
 * len will be updated to hold the array length
 *
 */
MTCR_API dev_info* mdevices_info(int mask, int *len);

/*
 *  * Get list of MST (Mellanox Software Tools) devices info records.
 *  * Return a dynamic allocated array of dev_info records.
 *  * len will be updated to hold the array length
 *  * Verbosity will decide whether to get all the Physical functions or not.
 */

MTCR_API dev_info* mdevices_info_v(int mask, int *len, int verbosity);

/*
 * Destroy the array of dev_info recored obtained by mdevices_info\() function
 *
 */
MTCR_API void mdevices_info_destroy(dev_info *dev_info, int len);

/*
 * Open Mellanox Software tools (mst) driver.
 * Return valid mfile ptr or 0 on failure
 */
MTCR_API mfile* mopend(const char *input_name, DType dtype);

/*
 * update mfile for gearbox device.
 * return 1=success , 0=fail
 */
int mopen_gearbox(char *input_name, mfile *mf);

/*
 * update gearbox device mfile with slave address and address width.
 */
void update_gearbox_mFile(mfile *mf, int isGb, int isGbManager);

/*
 * Open Mellanox Software tools (mst) driver.
 * Return valid mfile ptr or 0 on failure
 */
MTCR_API mfile* mopen_adv(const char *name, MType mtype);

/*
 * Open Mellanox Software tools (mst) driver. Device type=InfiniHost MType=MST_DEFAULT
 * Return valid mfile ptr or 0 on failure
 */
MTCR_API mfile* mopen(const char *name);

/*
 * Open Mellanox Software tools with uefi driver
 * Return valid mfile ptr or 0 on failure
 * NOTE: mfile will not conatin device data just context and uefi access function
 * supports only maccess_reg() function.
 */
MTCR_API mfile* mopen_fw_ctx(void *fw_cmd_context, void *fw_cmd_func, void* dma_func, void *extra_info);

/*
 * Close Mellanox driver
 * req. descriptor
 */
MTCR_API int mclose(mfile *mf);

/*
 * Accelerate device if possible.
 * When device is I2C master - overclock it
 */
MTCR_API void maccelerate(mfile *mf);

/*
 * Restore normal settings, if device was accelerated.
 */
MTCR_API void mrestore(mfile *mf);

/*
 * Read 4 bytes, return number of succ. read bytes or -1 on failure
 */
MTCR_API int mread4(mfile *mf, unsigned int offset, u_int32_t *value);

/*
 * Write 4 bytes, return number of succ. written bytes or -1 on failure
 */
MTCR_API int mwrite4(mfile *mf, unsigned int offset, u_int32_t value);

/*
 * Read a block of dwords, return number of succ. read bytes or -1 on failure
 * Works for any interface, but can be faster for interfaces where bursts
 * are supported (MTUSB, IB).
 * Data retrns in the same endianess of mread4/mwrite4
 */
MTCR_API int mread4_block(mfile *mf, unsigned int offset, u_int32_t *data, int byte_len);
MTCR_API int mwrite4_block(mfile *mf, unsigned int offset, u_int32_t *data, int byte_len);

/* read buffer as is without changing endians */
MTCR_API int mread_buffer(mfile *mf, unsigned int offset, u_int8_t *data, int byte_len);

/* Write buffer as is without changing endians */

MTCR_API int mwrite_buffer(mfile *mf, unsigned int offset, u_int8_t *data, int byte_len);

/*
 * Read up to 64 bytes, return number of succ. read bytes or -1 on failure
 */
MTCR_API int mread64(mfile *mf, unsigned int offset, void *data, int length);

/*
 * Write up to 64 bytes, return number of succ. written bytes or -1 on failure
 */
MTCR_API int mwrite64(mfile *mf, unsigned int offset, void *data, int length);

/*
 * Read up to 64 bytes, return number of succ. read bytes or -1 on failure
 */
MTCR_API int mread_i2cblock(mfile *mf, unsigned char i2c_slave, u_int8_t addr_width,
                            unsigned int offset, void *data, int length);

/*
 * Write up to 64 bytes, return number of succ. written bytes or -1 on failure
 */
MTCR_API int mwrite_i2cblock(mfile *mf, unsigned char i2c_slave, u_int8_t addr_width,
                             unsigned int offset, void *data, int length);

/*
 * Set a new value for i2c_slave
 * Return previous value
 */
MTCR_API unsigned char mset_i2c_slave(mfile *mf, unsigned char new_i2c_slave);
MTCR_API int mget_i2c_slave(mfile *mf, unsigned char *new_i2c_slave_p);


MTCR_API int mset_i2c_addr_width(mfile *mf, u_int8_t addr_width);
MTCR_API int mget_i2c_addr_width(mfile *mf, u_int8_t *addr_width);

MTCR_API int mget_mdevs_flags(mfile *mf, u_int32_t *devs_flags);
MTCR_API int mget_mdevs_type(mfile *mf, u_int32_t *mtype);
/*
 * Software reset the device.
 * Return 0 on success, <0 on failure.
 * Currently supported for IB device only.
 * Mellanox switch devices support this feature.
 * HCAs may not support this feature.
 */
MTCR_API int msw_reset(mfile *mf);

/*
 * reset the device.
 * Return 0 on success, <0 on failure.
 * Curently supported on 5th Generation HCAs.
 */
MTCR_API int mhca_reset(mfile *mf);

MTCR_API int mi2c_detect(mfile *mf, u_int8_t slv_arr[SLV_ADDRS_NUM]);

#define MTCR_MFT_2_7_0

MTCR_API int maccess_reg_mad(mfile *mf, u_int8_t *data);

MTCR_API int maccess_reg_cmdif(mfile *mf, reg_access_t reg_access, void *reg_data, u_int32_t cmd_type);

MTCR_API int maccess_reg(mfile     *mf,
                         u_int16_t reg_id,
                         maccess_reg_method_t reg_method,
                         void *reg_data,
                         u_int32_t reg_size,
                         u_int32_t r_size_reg,  // used when sending via icmd interface (how much data should be read back to the user)
                         u_int32_t w_size_reg,  // used when sending via icmd interface (how much data should be written to the scratchpad)
                                                // if you dont know what you are doing then r_size_reg = w_size_reg = your_register_size
                         int       *reg_status);

/**
 * Handles the send command procedure.
 * for completeness, but calling it is strongly advised against.
 * @param[in] dev   A pointer to a device context, previously
 *                  obtained by a call to <tt>gcif_open</tt>.
 * @return          One of the GCIF_STATUS_* values, or a raw
 *                  status value (as indicated in cr-space).
 * NOTE: when calling this function the caller needs to make
 *      sure device supports icmd.
 **/
MTCR_API int icmd_send_command(mfile    *mf,
                               int opcode,
                               void *data,
                               int data_size,
                               int skip_write);

/**
 * Clear the Tools-HCR semaphore. Use this when an application
 * that uses this library is not terminated cleanly, leaving the
 * semaphore in a locked state.
 * @param[in] dev   A pointer to a device context, previously
 *                  obtained by a call to <tt>gcif_open</tt>.
 * @return          One of the GCIF_STATUS_* values, or a raw
 *                  status value (as indicated in cr-space).
 * NOTE: when calling this function the caller needs to make
 *      sure device supports icmd.
 **/
MTCR_API int icmd_clear_semaphore(mfile *mf);

/*
 * send an inline command to the tools HCR
 * limitations:
 * command should not use mailbox
 * NOTE: when calling this function caller needs to make
 *       sure device support tools HCR
 */
MTCR_API int tools_cmdif_send_inline_cmd(mfile *mf,
                                         u_int64_t in_param,
                                         u_int64_t *out_param,
                                         u_int32_t input_modifier,
                                         u_int16_t opcode,
                                         u_int8_t opcode_modifier);

/*
 * send a mailbox command to the tools HCR
 * limitations:
 * i.e write data to mailbox execute command (op = opcode op_modifier= opcode_modifier) and read data back from mailbox
 * data_offs_in_mbox: offset(in bytes) to read and write data to and from mailbox should be quad word alligned.
 *  * NOTE: when calling this function caller needs to make
 *       sure device support tools HCR
 */
MTCR_API int tools_cmdif_send_mbox_command(mfile *mf,
                                           u_int32_t input_modifier,
                                           u_int16_t opcode,
                                           u_int8_t opcode_modifier,
                                           int data_offs_in_mbox,
                                           void *data,
                                           int data_size,
                                           int skip_write);

MTCR_API int tools_cmdif_unlock_semaphore(mfile *mf);

/*
 * returns the maximal allowed register size (in bytes)
 * according to the FW access method and access register method
 * or -1 if no restriction applicable
 *
 */
MTCR_API int mget_max_reg_size(mfile *mf, maccess_reg_method_t reg_method);
/*
 * translate virtual address to physical address
 * return physical address on success, or 0 on error
 */
MTCR_API unsigned long mvtop(mfile *mf, void *va);

MTCR_API const char* m_err2str(MError status);

MTCR_API int mvpd_read4(mfile *mf, unsigned int offset, u_int8_t value[4]);

MTCR_API int mvpd_write4(mfile *mf, unsigned int offset, u_int8_t value[4]);

MTCR_API int mib_smp_set(mfile *mf, u_int8_t *data, u_int16_t attr_id, u_int32_t attr_mod);
MTCR_API int mib_smp_get(mfile *mf, u_int8_t *data, u_int16_t attr_id, u_int32_t attr_mod);

MTCR_API int mget_vsec_supp(mfile *mf);
MTCR_API int supports_reg_access_gmp(mfile *mf, maccess_reg_method_t reg_method);

MTCR_API int mget_addr_space(mfile *mf);
MTCR_API int mset_addr_space(mfile *mf, int space);

MTCR_API int mclear_pci_semaphore(const char *name);

MTCR_API int get_dma_pages(mfile *mf, struct mtcr_page_info* page_info,
                           int page_amount);

MTCR_API int release_dma_pages(mfile *mf, int page_amount);

MTCR_API int read_dword_from_conf_space(u_int32_t offset, mfile *mf,
                                        struct mtcr_read_dword_from_config_space* read_config_space);

MTCR_API int MWRITE4_SEMAPHORE(mfile* mf, int offset, int value);

MTCR_API int MREAD4_SEMAPHORE(mfile* mf, int offset, u_int32_t* ptr);

MTCR_API int is_livefish_device(mfile *mf);

MTCR_API void set_increase_poll_time(int new_value);

MTCR_API int mcables_remote_operation_server_side(mfile* mf, u_int32_t address,
                                                  u_int32_t length, u_int8_t* data,
                                                  int remote_op);

MTCR_API int mcables_remote_operation_client_side(mfile* mf, u_int32_t address,
                                                  u_int32_t length, u_int8_t* data,
                                                  int remote_op);

MTCR_API int mlxcables_remote_operation_client_side(mfile* mf, const char* device_name,
                                                    char op, char flags);

#ifdef __cplusplus
}
#endif


#endif
