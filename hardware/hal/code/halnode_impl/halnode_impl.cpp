#include "halnode_impl.h"

int producer_start(int id)
{
    mem_mapinfo meminfo = { 0 };
    meminfo.sensor_id = id;

    int fd = open("/dev/netahal", O_RDWR);
    int ret = 0;

    if (fd < 0)
    {
        printf("can not open /dev/netahal\n");
        return -1;
    }

    ret = ioctl(fd,NETA_IO_PRODUCER , &meminfo);
    if (ret < 0)
    {
        close(fd);
        printf("ioctl failed(%d)\n", ret);
        return -1;
    }

   // close(fd);

    return ret;

}
int consumer_start(int id)
{
    mem_mapinfo meminfo = { 0 };
    meminfo.sensor_id = id;

    int fd = open("/dev/netahal", O_RDWR);
    int ret = 0;

    if (fd < 0)
    {
        printf("can not open /dev/netahal\n");
        return -1;
    }

    ret = ioctl(fd,NETA_IO_CONSUMER , &meminfo);
    if (ret < 0)
    {
        close(fd);
        printf("ioctl failed(%d)\n", ret);
        return -1;
    }

    //close(fd);

    return ret;

}
int set_producer_info(int id,sensor_info *i_sensor_info)
{
        mem_mapinfo meminfo = { 0 };
        meminfo.sensor_id = id;
        memcpy(&meminfo.sensorinfo,i_sensor_info,sizeof(sensor_info));

        int fd = open("/dev/netahal", O_RDWR);
        int ret = 0;

        if (fd < 0)
        {
                printf("can not open /dev/netahal\n");
                return -1;
        }

        ret = ioctl(fd,NETA_IO_SET_PRODUCER_INFO, &meminfo);
        if (ret < 0)
        {
                close(fd);
                printf("ioctl failed(%d)\n", ret);
                return -1;
        }

        close(fd);

    return ret;

}
int set_consumer_info(int id,int chanel_id, consumer_info *i_consumer_info)
{

        mem_mapinfo meminfo = { 0 };
        meminfo.sensor_id = id;
        meminfo.chanel_id = chanel_id;
        memcpy(&meminfo.consumerinfo,i_consumer_info,sizeof(consumer_info));


        int fd = open("/dev/netahal", O_RDWR);
        int ret = 0;

        if (fd < 0)
        {
                printf("can not open /dev/netahal\n");
                return -1;
        }

        ret = ioctl(fd,NETA_IO_SET_CONSUMER_INFO, &meminfo);
        if (ret < 0)
        {
                close(fd);
                printf("ioctl failed(%d)\n", ret);
                return -1;
        }

//        close(fd);

    return ret;

}


