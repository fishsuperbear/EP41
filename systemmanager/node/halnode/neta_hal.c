#include <linux/module.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/sched/signal.h>
#include <linux/delay.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>
#include <linux/timer.h>
#include "neta_klog.h"

// #pragma pack(8)

#ifndef s8
typedef signed char         s8;
#endif
#ifndef s16
typedef short               s16;
#endif
#ifndef s32
typedef int                 s32;
#endif
#ifndef s64
typedef long long           s64;
#endif
#ifndef u8
typedef unsigned char       u8;
#endif
#ifndef u16
typedef unsigned short      u16;
#endif
#ifndef u32
typedef unsigned int        u32;
#endif
#ifndef u64
typedef unsigned long long  u64;
#endif

#define NETA_MAX_ID 12
#define CONSUMER_SUM 6
#define PRODUCER_SUM 12

#define DRV_NAME "netahal"

static  int globalmem_major;

module_param(globalmem_major, int, S_IRUGO); //




/* Driver IOCTL codes*/

#define NETA_IO_DRV_MAGIC 'm'


#define NETA_IO_PRODUCER \
    _IOWR(NETA_IO_DRV_MAGIC, 1, mem_mapinfo)

#define NETA_IO_CONSUMER \
    _IOWR(NETA_IO_DRV_MAGIC, 2, mem_mapinfo)

#define NETA_IO_SET_PRODUCER_INFO \
    _IOWR(NETA_IO_DRV_MAGIC, 3, mem_mapinfo)

#define NETA_IO_SET_CONSUMER_INFO \
    _IOWR(NETA_IO_DRV_MAGIC, 4, mem_mapinfo)


typedef struct  data_info
{
	u32 bgpudata;
	char desc[32];	// enc, yuv422
} data_info;

typedef struct sensor_info
{
	int sensor_id;
	char desc[256];
} sensor_info;

typedef struct consumer_info{
	u32 flag; //1 start
	data_info data_info;
} consumer_info;

typedef struct producer_info{

	sensor_info sensor_info;
	u32 flag; //1 start
	consumer_info consumer_info[CONSUMER_SUM];
	u32 chanel_id;
} producerinfo;

typedef struct globalinfo_proc
{
	producerinfo producer_info[PRODUCER_SUM];
	bool producer_started;
	int consumer_started_sum;
	int producer_started_sum;

} globalinfo_proc;

typedef struct  {

	consumer_info consumer_info;
	sensor_info sensor_info;
	u32 sensor_id;
	u32 flags;  //0 producer 1 consumer
	u32 active;  //0 noactive  1 active
	u32 chanel_id; //0-5
} mem_mapinfo;

typedef struct globalmem_dev{
	struct class *char_class;
	struct cdev cdev;
	dev_t devno;

	mem_mapinfo mapinfo[NETA_MAX_ID]; // for store infomation
	unsigned int index;			//index of array
	unsigned int count;			//record information

	struct mutex mutex;

	wait_queue_head_t r_wait;
	wait_queue_head_t w_wait;
} globalmem_dev;
// #pragma pack()

struct globalmem_dev * globalmem_devp;

struct proc_dir_entry *netahal_dir,*globalinfo_file;

struct globalinfo_proc *global_proc_point;
static char *readbuf;

static int globalmem_open(struct inode *inode, struct file *filp)
{

	filp->private_data = kzalloc(sizeof(producerinfo),	GFP_KERNEL);
	NETA_KLOG_INFO("device open \n");
	return 0;
}

static int globalmem_release(struct inode *inode, struct file *filp)
{


	producerinfo *pcdev =filp->private_data;
	u32 tmp_sensor_id;
	u32 tmp_chanel_id;
	tmp_sensor_id = pcdev->sensor_info.sensor_id;
	tmp_chanel_id = pcdev->chanel_id;

	NETA_KLOG_INFO("into globalmem_release \n");
	NETA_KLOG_INFO("filp->private_date : %d \n",pcdev->sensor_info.sensor_id);
	if(pcdev->flag ==1)  //flag for producer down
	{
		mutex_lock(&globalmem_devp->mutex);
		globalmem_devp->mapinfo[pcdev->sensor_info.sensor_id].active = 0;
		mutex_unlock(&globalmem_devp->mutex);

		global_proc_point->producer_started_sum--;
		global_proc_point->producer_info[pcdev->sensor_info.sensor_id].flag =0;
		memset(&global_proc_point->producer_info[pcdev->sensor_info.sensor_id].sensor_info,0,sizeof(sensor_info));
	}

	if(pcdev->flag == 2) //flag for consumer down
	{
		memset(&global_proc_point->producer_info[tmp_sensor_id].consumer_info[tmp_chanel_id],0,sizeof(consumer_info));
		global_proc_point->consumer_started_sum--;
		global_proc_point->producer_info[tmp_sensor_id].consumer_info[tmp_chanel_id].flag =0;

	}

	if(global_proc_point->producer_started_sum == 0)
	{
		global_proc_point->producer_started = 0;
	}

	kfree(pcdev);
	NETA_KLOG_INFO("device close \n");
	return 0;
}
static int globalmem_free_map(struct globalmem_dev *dev, void __user *arg)
{
	return 0;
}

static int set_producer_info(struct file *filp, void __user *arg)
{
	mem_mapinfo set_producer_info;
	if (unlikely(copy_from_user(&set_producer_info, arg, sizeof(mem_mapinfo)))) {
			return -EFAULT;
	}
	memcpy(&global_proc_point->producer_info[set_producer_info.sensor_id].sensor_info,&set_producer_info.sensor_info,sizeof(sensor_info));
	global_proc_point->producer_info[set_producer_info.sensor_id].flag =1;
	return 0;
}
static int set_consumer_info(struct file *filp, void __user *arg)
{
	producerinfo *pcdev =filp->private_data;

	mem_mapinfo set_consumer_info;
	if (unlikely(copy_from_user(&set_consumer_info, arg, sizeof(mem_mapinfo)))) {
			return -EFAULT;
	}

	NETA_KLOG_INFO("chanel_id %d\n",set_consumer_info.chanel_id);
	NETA_KLOG_INFO("sensor_id %d   \n",set_consumer_info.sensor_id);

	pcdev->sensor_info.sensor_id = set_consumer_info.sensor_id;
	pcdev->chanel_id = set_consumer_info.chanel_id;
	pcdev->flag = 2;
	global_proc_point->consumer_started_sum++;

	//print_hex_dump_bytes(KERN_INFO" consumer info: ", DUMP_PREFIX_NONE, (char *)&set_consumer_info, sizeof(mem_mapinfo));

	//print_hex_dump(KERN_INFO, "consumer info: ", DUMP_PREFIX_NONE, 32, 4,
	//		    (char *)&set_consumer_info,
	//		    sizeof(mem_mapinfo),
	//		    true);
	memcpy(&global_proc_point->producer_info[set_consumer_info.sensor_id].consumer_info[set_consumer_info.chanel_id],\
	&set_consumer_info.consumer_info,sizeof(consumer_info));

	//global_proc_point->producer_info[set_consumer_info.sensor_id].chanel_id = set_consumer_info.chanel_id;

	NETA_KLOG_INFO("chanel_id %d\n",set_consumer_info.chanel_id);
	global_proc_point->producer_info[set_consumer_info.sensor_id].consumer_info[set_consumer_info.chanel_id].flag =1;
	return 0;
}

static long producer(struct file *filp,void __user *arg)
{
	mem_mapinfo map_producer;
	int ret=0;
	producerinfo *pcdev =filp->private_data;

	NETA_KLOG_INFO("INTO producer \n");

	//DECLARE_WAITQUEUE(wait, current);

	mutex_lock(&globalmem_devp->mutex);    // add lock
	//add_wait_queue(&globalmem_devp->w_wait, &wait);

	/*while(globalmem_devp->count == NETA_MAX_ID){ 	 //first check count

	__set_current_state(TASK_INTERRUPTIBLE);  	//set current task status as TASK_INTERRUPTIBLE

	mutex_unlock(&globalmem_devp->mutex);   //before schedule ,first unlock

	schedule();			//if count is ture ,schedule

	if(signal_pending(current)){
		ret = -ERESTARTSYS;
		goto out2;
	}
	mutex_lock(&globalmem_devp->mutex); //
	}*/



	if(signal_pending(current)){
		ret = -ERESTARTSYS;
		goto out;
	}
	globalmem_devp->count++;

	if (unlikely(copy_from_user(&map_producer, arg, sizeof(mem_mapinfo)))) {
			ret = -EFAULT;
			goto out;
	}
	pcdev->sensor_info.sensor_id = map_producer.sensor_id;
	pcdev->flag = 1;

	globalmem_devp->mapinfo[map_producer.sensor_id].active = 1;



	global_proc_point->producer_started =true;
	global_proc_point->producer_started_sum++;

	/*if (unlikely(copy_to_user(arg, &map_producer, sizeof(mem_mapinfo)))) {
			ret = -EFAULT;
			goto out;
	}*/



	wake_up_interruptible(&globalmem_devp->r_wait);   // wake up consumer


	NETA_KLOG_INFO("out producer \n");
out:
mutex_unlock(&globalmem_devp->mutex);
//out2:
//remove_wait_queue(&globalmem_devp->w_wait, &wait);
//set_current_state(TASK_RUNNING);
return ret;
}

static long consumer(struct file *filp,void __user *arg)
{
	mem_mapinfo map_consumer;
	long ret=0;
	DECLARE_WAITQUEUE(wait, current);

	NETA_KLOG_INFO("INTO consumer \n");

	mutex_lock(&globalmem_devp->mutex);
	add_wait_queue(&globalmem_devp->r_wait, &wait);

	if (unlikely(copy_from_user(&map_consumer, arg, sizeof(mem_mapinfo)))) {
		ret = -EFAULT;
		goto out;
	}

	while(globalmem_devp->mapinfo[map_consumer.sensor_id].active == 0){ 	 //first check count

	__set_current_state(TASK_INTERRUPTIBLE);  	//set current task status as TASK_INTERRUPTIBLE

	mutex_unlock(&globalmem_devp->mutex);

	schedule();			//if no producer, consumer schedule

	if(signal_pending(current)){
		ret = -ERESTARTSYS;
		goto out2;
	}
		mutex_lock(&globalmem_devp->mutex);
	}
	NETA_KLOG_INFO("out consumer \n");

out:
mutex_unlock(&globalmem_devp->mutex);
out2:
remove_wait_queue(&globalmem_devp->r_wait, &wait);
set_current_state(TASK_RUNNING);
return ret;

}


static long globalmem_ioctl (struct file *filp, unsigned int cmd, unsigned long arg)
{

	int err = -EINVAL;
	switch(cmd){
	case NETA_IO_PRODUCER:
		return producer(filp,(void __user *)arg);
	case NETA_IO_CONSUMER:
		return consumer(filp,(void __user *)arg);
	case NETA_IO_SET_PRODUCER_INFO:
		return set_producer_info(filp,(void __user *)arg);
	case NETA_IO_SET_CONSUMER_INFO:
		return set_consumer_info(filp,(void __user *)arg);
	default:
		break;
	}

	return err;
}

static int global_info_producerConsumer_show(struct seq_file *m,void *data)
{
	int i,j;
	seq_printf(m, "producer_active : %d \n",global_proc_point->producer_started);
	seq_printf(m, "producer_already_started : %d \n",global_proc_point->producer_started_sum);
	seq_printf(m, "consumer_already_started : %d \n",global_proc_point->consumer_started_sum);

	for(i=0; i < PRODUCER_SUM; i++)
	{
		if(global_proc_point->producer_info[i].flag == 1)
		{
			seq_printf(m, "sensor_id : %d\ndescription: %s \nstarted : %d\n", \
			i, \
			global_proc_point->producer_info[i].sensor_info.desc, \
			global_proc_point->producer_info[i].flag);
			for(j=0; j < CONSUMER_SUM; j++)
			{
				if(global_proc_point->producer_info[i].consumer_info[j].flag ==1 &&\
				global_proc_point->producer_info[i].consumer_info[j].data_info.bgpudata == 1){
					seq_printf(m," consumer[%d]:  CUDA\n", j);
				}
				if(global_proc_point->producer_info[i].consumer_info[j].flag ==1 &&\
				global_proc_point->producer_info[i].consumer_info[j].data_info.bgpudata == 0)
				{
					seq_printf(m," consumer[%d]:  ENC\n", j);
				}
			}
		}

	}
	return 0;
}

static int global_info_other_info_show(struct seq_file *m,void *data)
{
	int ret =-1;


	char filepath[] ="/var/log/halnode.txt";

	char *argv[] = {"/bin/sh", "-c", "lsof +D /dev |grep mgr > /var/log/halnode.txt",NULL};
	char *envp[] ={"HOME=/","TERM=linux","PATH=/sbin:/usr/sbin:/bin:/usr/bin",NULL};
	char *argv_neta[] = {"/bin/rm", "-f","/var/log/halnode.txt",NULL};

	mm_segment_t old_fs = get_fs();
	struct file* filnew;

	set_fs(KERNEL_DS);
	ret = call_usermodehelper(argv[0],argv,envp,UMH_WAIT_EXEC);
	set_fs(old_fs);

	// read file process
	msleep(1000);
	filnew = filp_open(filepath,O_RDWR,0644);
	if(IS_ERR(filnew))
	{
		NETA_KLOG_ERR("open halnode*.txt failed\n");
		return -1;
	}
	kernel_read(filnew, readbuf,1024,NULL);
	NETA_KLOG_INFO("readbuf :%s \n", readbuf);
	seq_printf(m, "\nRESOURCE GREP MGR : \n%s\n",readbuf);
	filp_close(filnew,NULL);

	//delete deadline file
	msleep(100);
	call_usermodehelper(argv_neta[0],argv_neta,envp,UMH_WAIT_EXEC);

	NETA_KLOG_INFO("out call_usermodehelper %x \n", ret);
	return ret;

}

static int global_info_proc_show(struct seq_file *m,void *date)
{
	global_info_producerConsumer_show(m,date);
	global_info_other_info_show(m,date);
	return 0;
}
static int global_info_proc_open(struct inode *i_pinode, struct file *i_pfile)
{

	return single_open(i_pfile, global_info_proc_show, NULL);

}
static int global_info_proc_release(struct inode *i_pinode, struct file *i_pfile)
{
    return single_release(i_pinode, i_pfile);
}

static ssize_t global_info_proc_read(struct file *i_pfile, char __user *io_pbuf, size_t i_count, loff_t *io_ppos)
{

    return seq_read(i_pfile, io_pbuf, i_count, io_ppos);
}


static const struct file_operations globalmem_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = globalmem_ioctl,
	.open = globalmem_open,
	.release = globalmem_release,

};

static const struct proc_ops globalfile_proc_ops = {
    .proc_open    = global_info_proc_open,
    .proc_read    = global_info_proc_read,
    .proc_write   = NULL,
    .proc_release = global_info_proc_release,
};



static void globalmem_setup_cdev(struct globalmem_dev *dev, int index)
{
	int err;

	alloc_chrdev_region(&globalmem_devp->devno, 0, 1, DRV_NAME);

	globalmem_major = MAJOR(globalmem_devp->devno);

	//dev->cdev = cdev_alloc();

	cdev_init(&dev->cdev, &globalmem_fops);
	dev->cdev.owner = THIS_MODULE;
	err = cdev_add(&dev->cdev, globalmem_devp->devno, 1);

	dev->char_class = class_create(THIS_MODULE, DRV_NAME);
    device_create(dev->char_class, ((void *)0), globalmem_devp->devno, ((void *)0), DRV_NAME);


	if (err) {
		NETA_KLOG_ERR("Error %d adding globalmem %d",err, index);
	}
}


int globalmem_init (void)
{
	globalmem_devp = kzalloc (sizeof(struct globalmem_dev),	GFP_KERNEL);   //alloc memory
	if (!globalmem_devp){
		return -1;
	}
	global_proc_point = kzalloc (sizeof(struct globalinfo_proc),GFP_KERNEL);
	if (!global_proc_point){
		return -1;
	}
	readbuf = (char *)kzalloc(sizeof(char)*1024,GFP_KERNEL);

	globalmem_setup_cdev(globalmem_devp , 1);

	memset(globalmem_devp->mapinfo, 0, NETA_MAX_ID*sizeof(mem_mapinfo));
	globalmem_devp->index = 0;
	globalmem_devp->count = 0;
	mutex_init(&globalmem_devp->mutex);	 	//init metux and work_queue
	init_waitqueue_head(&globalmem_devp->r_wait);
	init_waitqueue_head(&globalmem_devp->w_wait);

	global_proc_point->consumer_started_sum = 0;
	global_proc_point->producer_started_sum = 0;

	netahal_dir = proc_mkdir("netahal",NULL);
	globalinfo_file = proc_create("global_info",0444, netahal_dir, &globalfile_proc_ops);

	return 0;
}

 void globalmem_exit(void)
{

	device_destroy(globalmem_devp->char_class, globalmem_devp->devno);
    class_destroy(globalmem_devp->char_class);

	cdev_del(&globalmem_devp->cdev);

	unregister_chrdev_region(MKDEV(globalmem_major, 0 ), 1);

	remove_proc_entry("global_info",netahal_dir);
	remove_proc_entry("netahal",NULL);

    mutex_destroy(&globalmem_devp->mutex);

	kfree(globalmem_devp);
	kfree(global_proc_point);
	kfree(readbuf);
}

//module_init(globalmem_init);
//module_exit(globalmem_exit);

MODULE_AUTHOR("neta_zhaowenqiang");
MODULE_LICENSE("GPL v2");
