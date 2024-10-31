#ifndef HALNODE_H
#define HALNODE_H

#ifdef __cplusplus
extern "C" {
#endif

#define CONSUMER_SUM 6

//#pragma pack(8)

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


typedef struct  data_info
{
	u32 bgpudata;    // 0 enc  1 CUDA
	char desc[32];	// enc, yuv422
}data_info;

typedef struct sensor_info
{
	int sensor_id;
	char desc[256];
}sensor_info;

typedef struct consumer_info{

	u32 flag; //1 start
	data_info data_info;
}consumer_info;



// #pragma pack()

	
// i_sensorid begin from 0
int producer_start(int i_sensorid); 
// i_sensorid begin from 0          
int consumer_start(int i_sensorid); 
// set producer infomation interface
int set_producer_info(int id,sensor_info *i_sensor_info);
//set producer information interface
int set_consumer_info(int id, int chanel_id,consumer_info *i_consumer_info);

#ifdef __cplusplus
}
#endif

#endif
