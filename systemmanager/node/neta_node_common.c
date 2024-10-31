#include "neta_node.h"

char* para_logmode = "nolock";

ssize_t	neta_procops_read_default(struct file *i_pfile, char __user *io_pbuf, size_t i_count, loff_t *io_ppos)
{
    return seq_read(i_pfile, io_pbuf, i_count, io_ppos);
}

ssize_t	neta_procops_write_dummy(struct file *i_pfile, const char __user *i_pbuf, size_t i_count, loff_t *io_ppos)
{
    return 0;
}

loff_t neta_procops_lseek_default(struct file *i_pfile, loff_t i_offset, int i_whence)
{
    return seq_lseek(i_pfile, i_offset, i_whence);
}

int neta_procops_release_default(struct inode *i_pinode, struct file *i_pfile)
{
    return single_release(i_pinode, i_pfile);
}

int neta_fileops_open_dummy(struct inode *i_pinode, struct file *i_pfile)
{
    // put your private data pointer to i_pfile->private_data
    return 0;
}

ssize_t neta_fileops_read_dummy(struct file *i_pfile, char __user *io_pbuf, size_t i_count, loff_t *io_ppos)
{
    return 0;
}

ssize_t neta_fileops_write_dummy(struct file *i_pfile, const char __user *i_pbuf, size_t i_count, loff_t *io_ppos)
{
    return 0;
}

int neta_fileops_release_dummy(struct inode *i_pinode, struct file *i_pfile)
{
    // put your private data pointer to i_pfile->private_data
    return 0;
}

loff_t neta_fileops_llseek_notsupport(struct file *i_pfile, loff_t i_offset, int i_whence)
{
    // whence like SEEK_SET/SEEK_CUR/SEEK_END

    /*
    * -ESPIPE means not support llseek, you set llseek ops to NULL to 
    * use system default llseek.
    */
    return -ESPIPE;
}

struct sg_table *neta_dmabufops_mapdmabuf_notsupport(struct dma_buf_attachment *i_pattachment, enum dma_data_direction i_dir)
{
	return NULL;
}

void neta_dmabufops_unmapdmabuf_notsupport(struct dma_buf_attachment *i_pattachment, struct sg_table *i_ptable, enum dma_data_direction i_dir)
{

}

void* neta_dmabufops_map_notsupport(struct dma_buf *i_pdmabuf, unsigned long i_page_num)
{
	return NULL;
}
