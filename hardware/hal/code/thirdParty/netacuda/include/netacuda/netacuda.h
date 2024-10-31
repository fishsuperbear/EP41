#ifndef NETA_CUDA_H__
#define NETA_CUDA_H__

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

/**
 * @brief Initializes the MPS client.
 *
 * This function initializes the MPS client and prepares it for communication with the MPS server. The MPS client is used to enable concurrent execution of multiple CUDA contexts on a single GPU device.
 *
 * @return Returns a boolean value indicating the success of the initialization. 'true' indicates successful initialization, while 'false' indicates failure.
 *
 * @note This function must be called before creating a CUDA stream and after setting the device.
 */
// bool neta_init_mps_context(CUcontext &cuda_context);
s32 neta_cuda_init(u32 i_enablemps);

#endif //NETA_CUDA_H__