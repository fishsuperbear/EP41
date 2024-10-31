#ifndef CODEC_ORIN_IEP_IMAGE_ENCODER_API_H_
#define CODEC_ORIN_IEP_IMAGE_ENCODER_API_H_

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
typedef unsigned char uint8_t;
#endif

int Init(const char* config_file, void **context, bool nvstream_mode, void* cbs);
int Deinit(void *context);

int Process(void const *in, uint8_t **out, int* out_size, int *frame_type, void *context);
int GetEncoderParam(void *context, int param_type, void **param);
int SetEncoderParam(void *context, int param_type, void *param);

#ifdef __cplusplus
}
#endif

#endif // CODEC_ORIN_IEP_IMAGE_ENCODER_API_H_