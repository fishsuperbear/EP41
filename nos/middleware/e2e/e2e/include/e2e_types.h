#ifndef E2E_TYPES_H_
#define E2E_TYPES_H_
#include <bits/stdint-uintn.h>

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef TRUE
#define TRUE 1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef BOOL
#define BOOL int
#endif

#ifndef NULL
#define NULL ((void*)0)
#endif

typedef BOOL boolean;

typedef enum {
    E2E_E_OK = 0x00u,
    E2E_E_INPUTERR_NULL = 0x13u,   // At least one pointer parameter is a NULL pointer
    E2E_E_INPUTERR_WRONG = 0x17u,  // At least one input parameter is erroneous, e.g. out of range
    E2E_E_INTERR = 0x19u,          // An internal library error has occurred (e.g. error detected by program flow monitoring, violated invariant or postcondition
    E2E_E_WRONGSTATE = 0x1Au,      // Function executed in wrong state
    E2E_E_INVALID = 0xFFu
} Std_ReturnType;

/* [SWS_Std_00028] */
typedef enum {
    E_OK = 0x00u,                  // No specific error to be injected
    E_SAFETY_INVALID_REP = 0x01u,  // Repeat the last used sequence number
    E_SAFETY_INVALID_SEQ = 0x02u,  // Generate a deliberately wrong CRC
    E_SAFETY_INVALID_ERR = 0x03u   // Use a wrong sequence number
} Std_TransformerForwardCode;      // DRAFT

/* [SWS_Std_00015] */
typedef struct {
    uint16_t vendorID;
    uint16_t moduleID;
    uint8_t sw_major_version;
    uint8_t sw_minor_version;
    uint8_t sw_patch_version;
}Std_VersionInfoType;

#ifdef __cplusplus
}
#endif
#endif