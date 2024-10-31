#ifndef HW_HAL_API_H
#define HW_HAL_API_H

#include "hw_platform.h"
#include "hw_tag_and_version.h"

/*
* We do not include hw_global_devtype.h so that some implement code 
* will NOT be affected by global devtype version.
*/

__BEGIN_DECLS

struct hw_module_t;
struct hw_device_t;

/*
* Name of the xx_module_t symbol in the implemented module code.
* Defined in the implemented module code.
*/
#define HAL_MODULE_INFO_SYM         HMI

/*
* Name string of the xx_module_t symbol in the implemented module code.
* We use the string to find the symbol of the implemented module code.
*/
#define HAL_MODULE_INFO_SYM_AS_STR  "HMI"

typedef struct hw_module_privdata_t {
    /*
    * Module's dso, the return value of dll open operation.
    * Used to do dll close operation.
    * Always is set to NULL on the module implemented side.
    */
    void* dso;

    /*
    * The module so can set it by init function io_ppvoid.
    * Only used when call deinit function.
    * Owned by the module itself. Hal code do NOT change it.
    */
    void* pvoid;
} hw_module_privdata_t;

/*
* The private functions which should only be called by hardware hal code
* directly. The module users can NOT call them.
* All of the function in the structure should be implemented by the module 
* implementation.
*/
typedef struct hw_module_privapi_t {
    /*
    * Called by hw_module_get function.
    * io_ppvoid is the address of pvoid, to receive the pvoid pointer from 
    * module. Currently we may not use the pvoid, reserved for future use.
    * return 0 means success.
    * return <0 means already init.
    * When it has already init, it cannot return 0, it should return <0.
    */
    s32(*init)(void** io_ppvoid);

    /*
    * Called by hw_module_get function after calling init.
    * The module implement should check whether the module can support the 
    * specific input version of the device.
    * When we support the specific version device api, we need to set the same
    * device api version to the output hw_device_t when you use 
    * hw_module_device_get to get the hw_device_t.
    * 
    * @return: 0 means support. <0 means NOT support.
    */
    s32(*check_device_api_version)(u32 i_device_api_version);

    /*
    * Currently we may not use it. Remained for future use.
    * Called when we need to trigger log output the unmasked log in the
    * circumstance like just finished initing the log system output.
    *
    * @return: 0 means all of the unmasked log has been output or there's
    * no unmasked log.
    * -1 means there's unmasked log left in the inner unmasked log buffer
    * which can not output now due to log output fail.
    */
    s32(*trigger_check_unmasklogleft)();

    /*
    * You should call hw_module_device_get instead.
    * Define it in the common module structure so that the module
    * users can do the device_get operation without know the
    * specific module structure define.
    */
    s32(*device_get)(struct hw_module_t* i_pmodule, void* i_param, struct hw_device_t** io_ppdevice);
    /*
    * You should call hw_module_device_put instead.
    * The pair operation of device_get.
    */
    s32(*device_put)(struct hw_module_t* i_pmodule, struct hw_device_t* i_pdevice);
} hw_module_privapi_t;

/*
* Every hardware module must begin with hw_module_t followed by module specific 
* information.
*/
typedef struct hw_module_t {
    /*
    * Set by module implementation.
    * Tag must be initialized to HARDWARE_MODULE_TAG.
    */
    u32 tag;

    /*
    * Set by module implementation.
    * It is the version of the HAL module interface. This is meant to
    * version the hw_module_t, hw_module_methods_t, and hw_device_t
    * structures and definitions.
    * For example, version 1.0 could be represented as 0x0100. This format
    * implies that versions 0x0100-0x01ff are all API-compatible.
    * Presently, 0 is the only valid value. Once update, the affect is 
    * global.
    * Should be HARDWARE_HAL_API_VERSION.
    */
    u16 hal_api_version;
    /*
    * Set by module implementation.
    * It is the version of the devicetype enum list. 
    * When release the version, all devicetype enum of the released 
    * version should remain the same until new version release. You can 
    * add new devicetype value, but they should NOT equal to any exist 
    * devicetype value of the released Xb version.
    * For example, version 1.0 could be represented as 0x0100. This format
    * implies that versions 0x0100-0x01ff are all devicetype-compatible.
    * Use macro HW_MAKEV_GLOBAL_DEVICETYPE_VERSION.
    * Should be HW_GLOBAL_DEVTYPE_VERSION. Properly defined in 
    * hw_global_devtype_vx.x.h.
    */
    u16 global_devicetype_version;

    /*
    * Set by module implementation.
    * See HW_DEVICETYPE in hw_global_devtype_vx.x.h.
    * One devicetype correspondent to one hw_device_t.
    * The global devicetype enum list version is within global_devicetype_version.
    * See global_devicetype_version note for details.
    */
    u16 devicetype;

    /*
    * Set by module implementation.
    * When the devicetype is certain, the module_id is the Identifier of module.
    * Of course, you can use different module using different module_id
    * to serve the same devicetype device.
    */
    u16 module_id;

    /*
    * Set by module implementation.
    * The magic number of device type.
    * To check version match only.
    * Use hw_global_devtype_magic_get to get.
    */
    u32 devtype_magic;

    /*
    * The module id list version of the specific device type.
    * For example, version 1.0 could be represented as 0x0100. This format
    * implies that versions 0x0100-0x01ff are all moduleid-compatible.
    * Use macro HW_MAKEV_DEVICE_MODULEID_VERSION.
    * Should be HW_DEVICE_MODULEID_VERSION, properly defined in 
    * hw_xx_moduleid_vx.x.h.
    */
    u16 device_moduleid_version;
    /*
    * The version of specific devicetype specific device_module_id.
    * The module API version should include a major and a minor component.
    * For example, version 1.0 could be represented as 0x0100. This format
    * implies that versions 0x0100-0x01ff are all API-compatible.
    * Use macro HW_MAKEV_MODULE_API_VERSION.
    * Should be HW_MODULE_API_VERSION, properly defined in 
    * hw_xxx_xxx.h the specific module header file.
    */
    u16 module_api_version;

    /*
    * Set by module implementation.
    * The magic number of module id list of the specific device type.
    * To check version match only.
    */
    u32 devmoduleid_magic;

    /*
    * Description of this module, CANNOT be NULL, CAN be "".
    * Set by module implement.
    */
    const char* description;

    struct hw_module_privdata_t privdata;

    struct hw_module_privapi_t privapi;

#if (HW_PLAT_POINTER_BITNUM == 64)
    /*
    * For padding and reserved use. Padding to 128 bytes.
    * The following is the padding example code.
    */
    u64 reserved[5];    // 128 bytes
#endif

} hw_module_t;

#if (HW_PLAT_POINTER_BITNUM == 64)
STATIC_ASSERT(sizeof(hw_module_t) == 128);
#endif

/*
* Every device data structure must begin with hw_device_t
* followed by module specific public methods and attributes.
*/
typedef struct hw_device_t {
    /** tag must be initialized to HARDWARE_DEVICE_TAG */
    u32 tag;

    /*
    * Set by module implementation.
    * The module implement should store it down when the system calls the 
    * init callback function of hw_module_privapi_t.
    * And when the system calls the device_get function, the module 
    * implement should set the device_api_version as the same value just 
    * written down in the init callback function of hw_module_privapi_t.
    * Version of the device api. Once the code is certain, all of the 
    * xx_device_t instance's device_api_version should be the same.
    * Once the code release, the specific version's xx_device_t api should 
    * NOT be changed later.
    *
    * One module can support multiple devices with different versions. This
    * can be useful when we can update hal or device one by one in a 
    * compatible way using device or module version check.
    * Use HW_MAKEV_DEVICE_API_VERSION to generate the device api version of 
    * specific device. Properly defined like HW_XXX_API_VERSION in file like
    * hw_xx_versionl.h.
    */
    u32 device_api_version;

    /** reference to the module this device belongs to */
    struct hw_module_t* pmodule;

    /** padding reserved for future use */
#if (HW_PLAT_POINTER_BITNUM == 64)
    u64 reserved[(64-16)/8];
#endif

} hw_device_t;

#if (HW_PLAT_POINTER_BITNUM == 64)
STATIC_ASSERT(sizeof(hw_device_t) == 64);
#endif

/*
* Called by hardware hal users.
* Get the xx_module_t structure by so library of the input i_modulesoname.
* It will automatically called the init function set by module so library.
*
* @return: 0 means success, <0 means other error.
*/
s32 hw_module_get(const char* i_modulesoname, struct hw_module_t** o_ppmodule);

/*
* Called by hardware hal users.
* 
* @return: 0 means success, <0 means other error.
*/
s32 hw_module_put(struct hw_module_t* i_pmodule);

/*
* The function will check the device_api_version of the output io_ppdevice.
* It should be the same as the device current version.
*/
s32 hw_module_device_get(struct hw_module_t* i_pmodule, void* i_param, struct hw_device_t** o_ppdevice);

s32 hw_module_device_put(struct hw_module_t* i_pmodule, struct hw_device_t* i_pdevice);

__END_DECLS

#endif
