#include <linux/module.h>
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/kernel.h>
#include "neta_klog.h"

#define HALLOG_CTRL_LEVEL_FATAL          8
#define HALLOG_CTRL_LEVEL_UNMASK         6
#define HALLOG_CTRL_LEVEL_ERROR          5
#define HALLOG_CTRL_LEVEL_WARNING        4
#define HALLOG_CTRL_LEVEL_INFO           3
#define HALLOG_CTRL_LEVEL_TRACE          2
#define HALLOG_CTRL_LEVEL_DEBUG          1

#define LOG_LEVEL_BUF_SIZE         64

static struct proc_dir_entry *log_dir;
static struct proc_dir_entry *hw_hal_file;
static struct proc_dir_entry *hw_nvmedia_file;
static struct proc_dir_entry *hal_camera_file;
static struct proc_dir_entry *global_file;
extern struct proc_dir_entry *netahal_dir;

struct loglevel_buffer
{
    char hw_hal_loglevel[LOG_LEVEL_BUF_SIZE];
    char hw_nvmedia_loglevel[LOG_LEVEL_BUF_SIZE];
    char hal_camera_loglevel[LOG_LEVEL_BUF_SIZE];
    char global_loglevel[LOG_LEVEL_BUF_SIZE];
};

 static struct loglevel_buffer loglevel = {
    .hw_hal_loglevel = "5",
    .hw_nvmedia_loglevel = "5",
    .hal_camera_loglevel = "5",
    .global_loglevel = "5"
};

static int hw_hal_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%s\n", loglevel.hw_hal_loglevel);

    return 0;
}

static int hw_nvmedia_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%s\n", loglevel.hw_nvmedia_loglevel);

    return 0;
}

static int hal_camera_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%s\n", loglevel.hal_camera_loglevel);

    return 0;
}
static int global_show(struct seq_file *m, void *v)
{
    seq_printf(m, "%s\n", loglevel.global_loglevel);

    return 0;
}

static int hw_hal_open(struct inode *inode, struct file *file)
{
    return single_open(file, hw_hal_show, NULL);
}

static int hw_nvmedia_open(struct inode *inode, struct file *file)
{
    return single_open(file, hw_nvmedia_show, NULL);
}

static int hal_camera_open(struct inode *inode, struct file *file)
{
    return single_open(file, hal_camera_show, NULL);
}

static int global_open(struct inode *inode, struct file *file)
{
    return single_open(file, global_show, NULL);
}

static ssize_t hallog_ctrl_read(struct file *file, char __user *buffer, size_t count, loff_t *offset)
{
    return seq_read(file, buffer, count, offset);
};

static ssize_t hw_hal_write(struct file *file, const char __user *buffer,
                         size_t count, loff_t *offset)
{
    char input_data[LOG_LEVEL_BUF_SIZE];
    int log_level;

    if (count >= LOG_LEVEL_BUF_SIZE) {
        return -1;
    }

    if (copy_from_user(input_data, buffer, count)) {
        return -1;
    }
    input_data[count] = '\0';
    if (kstrtoint(input_data, 10, &log_level)) {
        return -1;
    }

    if (log_level > 5) {
        NETA_KLOG_ERR("input argument error\n");
        return -1;
    }

    if (copy_from_user(loglevel.hw_hal_loglevel, buffer, count)) {
        return -1;
    }

    return count;
}

static ssize_t hw_nvmedia_write(struct file *file, const char __user *buffer,
                         size_t count, loff_t *offset)
{
    char input_data[LOG_LEVEL_BUF_SIZE];
    int log_level;

    if (count >= LOG_LEVEL_BUF_SIZE) {
        return -1;
    }

    if (copy_from_user(input_data, buffer, count)) {
        return -1;
    }
    input_data[count] = '\0';
    if (kstrtoint(input_data, 10, &log_level)) {
        return -1;
    }

    if (log_level > 5) {
        NETA_KLOG_ERR("input argument error\n");
        return -1;
    }

    if (copy_from_user(loglevel.hw_nvmedia_loglevel, buffer, count)) {
        return -1;
    }

    return count;
}

static ssize_t hal_camera_write(struct file *file, const char __user *buffer,
                         size_t count, loff_t *offset)
{
    char input_data[LOG_LEVEL_BUF_SIZE];
    int log_level;

    if (count >= LOG_LEVEL_BUF_SIZE) {
        return -1;
    }

    if (copy_from_user(input_data, buffer, count)) {
        return -1;
    }
    input_data[count] = '\0';
    if (kstrtoint(input_data, 10, &log_level)) {
        return -1;
    }

    if (log_level > 5) {
        NETA_KLOG_ERR("input argument error\n");
        return -1;
    }

    if (copy_from_user(loglevel.hal_camera_loglevel, buffer, count)) {
        return -1;
    }

    return count;
}

static ssize_t global_write(struct file *file, const char __user *buffer,
                         size_t count, loff_t *offset)
{
    char input_data[LOG_LEVEL_BUF_SIZE];
    int log_level;

    if (count >= LOG_LEVEL_BUF_SIZE) {
        return -1;
    }

    if (copy_from_user(input_data, buffer, count)) {
        return -1;
    }
    input_data[count] = '\0';
    if (kstrtoint(input_data, 10, &log_level)) {
        return -1;
    }

    if (log_level > 5) {
        NETA_KLOG_ERR("input argument error\n");
        return -1;
    }

    if (copy_from_user(loglevel.global_loglevel, buffer, count)) {
        return -1;
    }
    if (copy_from_user(loglevel.hw_hal_loglevel, buffer, count)) {
        return -1;
    }
    if (copy_from_user(loglevel.hw_nvmedia_loglevel, buffer, count)) {
        return -1;
    }
    if (copy_from_user(loglevel.hal_camera_loglevel, buffer, count)) {
        return -1;
    }

    return count;
}

static const struct proc_ops hw_hal_proc_ops = {
    .proc_read = hallog_ctrl_read,
    .proc_write = hw_hal_write,
    .proc_open = hw_hal_open,
    .proc_release = single_release,
    .proc_ioctl = NULL,
};

static const struct proc_ops hw_nvmedia_proc_ops = {
    .proc_read = hallog_ctrl_read,
    .proc_write = hw_nvmedia_write,
    .proc_open = hw_nvmedia_open,
    .proc_release = single_release,
    .proc_ioctl = NULL,
};

static const struct proc_ops hal_camera_proc_ops = {
    .proc_read = hallog_ctrl_read,
    .proc_write = hal_camera_write,
    .proc_open = hal_camera_open,
    .proc_release = single_release,
    .proc_ioctl = NULL,
};

static const struct proc_ops global_proc_ops = {
    .proc_read = NULL,
    .proc_write = global_write,
    .proc_open = global_open,
    .proc_release = single_release,
    .proc_ioctl = NULL,
};

int hallog_ctrl_init(void)
{
    log_dir = proc_mkdir("hallog_ctrl", netahal_dir);
    if (!log_dir) {
        NETA_KLOG_ERR("Failed to create /proc/netahal/hallog_ctrl/\n");
        return -1;
    }

    hw_hal_file = proc_create("hw_hal", 0444, log_dir, &hw_hal_proc_ops);
    if (!hw_hal_file) {
        NETA_KLOG_ERR("Failed to create /proc/netahal/hallog_ctrl/hw_hal\n");
        return -2;
    }

    hw_nvmedia_file = proc_create("hw_nvmedia", 0444, log_dir, &hw_nvmedia_proc_ops);
    if (!hw_nvmedia_file) {
        NETA_KLOG_ERR("Failed to create /proc/netahal/hallog_ctrl/hw_nvmedia\n");
        return -2;
    }

    hal_camera_file = proc_create("hal_camera", 0444, log_dir, &hal_camera_proc_ops);
    if (!hal_camera_file) {
        NETA_KLOG_ERR("Failed to create /proc/netahal/hallog_ctrl/hal_camera\n");
        return -2;
    }

    global_file = proc_create("global", 0444, log_dir, &global_proc_ops);
    if (!global_file) {
        NETA_KLOG_ERR("Failed to create /proc/netahal/hallog_ctrl/global\n");
        return -2;
    }

    return 0;
};

void hallog_ctrl_exit(void)
{
    remove_proc_entry("hw_hal",log_dir);
    remove_proc_entry("hw_nvmedia",log_dir);
    remove_proc_entry("hal_camera",log_dir);
    remove_proc_entry("global",log_dir);
    remove_proc_entry("hallog_ctrl",netahal_dir);
    NETA_KLOG_INFO("/proc/netahal/hallog_ctrl removed\n");
};

MODULE_LICENSE("GPL");
MODULE_AUTHOR("neta_yuchaoxiong");
MODULE_DESCRIPTION("Log Level Control Module");

