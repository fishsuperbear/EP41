#ifndef NETA_NODE_COMMON_H
#define NETA_NODE_COMMON_H

#include "neta_klog.h"

ssize_t	neta_procops_read_default(struct file *i_pfile, char __user *io_pbuf, size_t i_count, loff_t *io_ppos);
ssize_t	neta_procops_write_dummy(struct file *i_pfile, const char __user *i_pbuf, size_t i_count, loff_t *io_ppos);
loff_t neta_procops_lseek_default(struct file *i_pfile, loff_t i_offset, int i_whence);
int neta_procops_release_default(struct inode *i_pinode, struct file *i_pfile);

int neta_fileops_open_dummy(struct inode *i_pinode, struct file *i_pfile);
ssize_t neta_fileops_read_dummy(struct file *i_pfile, char __user *io_pbuf, size_t i_count, loff_t *io_ppos);
ssize_t neta_fileops_write_dummy(struct file *i_pfile, const char __user *i_pbuf, size_t i_count, loff_t *io_ppos);
int neta_fileops_release_dummy(struct inode *i_pinode, struct file *i_pfile);
loff_t neta_fileops_llseek_notsupport(struct file *i_pfile, loff_t i_offset, int i_whence);

struct sg_table *neta_dmabufops_mapdmabuf_notsupport(struct dma_buf_attachment *i_pattachment, enum dma_data_direction i_dir);
void neta_dmabufops_unmapdmabuf_notsupport(struct dma_buf_attachment *i_pattachment, struct sg_table *i_ptable, enum dma_data_direction i_dir);
void* neta_dmabufops_map_notsupport(struct dma_buf *i_pdmabuf, unsigned long i_page_num);

/*
* For insmod parameter input use.
*/
extern char* para_logmode;

#endif
