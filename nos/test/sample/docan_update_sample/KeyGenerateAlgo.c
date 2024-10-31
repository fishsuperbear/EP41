#include <windows.h>
#include <stdint.h>
#include "KeyGenAlgoInterfaceEx.h"

#define   DIAG_MDC_SECURITY_ACCESS_APP_MASK              (0x23AEBEFD)
#define   DIAG_MDC_SECURITY_ACCESS_BOOT_MASK             (0xAB854A17)

#define   DIAG_CONTI_LRR_SECURITY_ACCESS_APP_MASK        (0x7878824A)
#define   DIAG_CONTI_LRR_SECURITY_ACCESS_BOOT_MASK       (0x7777834B)

#define   DIAG_CHUHANG_LRR_SECURITY_ACCESS_APP_MASK      (0x9AB07B6A)
#define   DIAG_CHUHANG_LRR_SECURITY_ACCESS_BOOT_MASK     (0x526A3583)

#define   DIAG_HESAI_LIDAR_SECURITY_ACCESS_APP_MASK      (0x5F2F6BD0)
#define   DIAG_HESAI_LIDAR_SECURITY_ACCESS_BOOT_MASK     (0X9C5A115D)

#define   DIAG_SSTK_SRR_SECURITY_ACCESS_APP_MASK         (0x51D96F09)
#define   DIAG_SSTK_SRR_SECURITY_ACCESS_BOOT_MASK        (0x71887503)

#define   DIAG_LISHENG_ADCU_SECURITY_ACCESS_APP_MASK     (0xF5B72931)
#define   DIAG_LISHENG_ADCU_SECURITY_ACCESS_BOOT_MASK    (0xF312E34B)

#define   DIAG_HIRAIN_FLC_SECURITY_ACCESS_APP_MASK       (0x23001537)
#define   DIAG_HIRAIN_FLC_SECURITY_ACCESS_BOOT_MASK      (0x00847016)

#define   DIAG_HUACE_IMU_SECURITY_ACCESS_APP_MASK        (0x51A57C08)
#define   DIAG_HUACE_IMU_SECURITY_ACCESS_BOOT_MASK       (0x2140242B)

#define   DIAG_USSC_SECURITY_ACCESS_APP_MASK       (0x3CF27981)
#define   DIAG_USSC_SECURITY_ACCESS_BOOT_MASK      (0x1C6D520B)

#define   DIAG_HIRAIN_ADCS2_SECURITY_ACCESS_APP_MASK       (0xF5B72931)
#define   DIAG_HIRAIN_ADCS2_SECURITY_ACCESS_BOOT_MASK      (0xF312E34B)


unsigned int GetKeyAppLevel(unsigned int seed, unsigned int APP_MASK)
{
	unsigned int ret = 0;
	unsigned int tmpseed = seed;
	unsigned int key_1 = tmpseed ^ APP_MASK;
	unsigned int seed_2 = tmpseed;
	if (seed == 0) {
		return 0;
	}

	seed_2 = (seed_2 & 0x55555555) << 1 ^ (seed_2 & 0xAAAAAAAA) >> 1;
	seed_2 = (seed_2 ^ 0x33333333) << 2 ^ (seed_2 ^ 0xCCCCCCCC) >> 2;
	seed_2 = (seed_2 & 0x0F0F0F0F) << 4 ^ (seed_2 & 0xF0F0F0F0) >> 4;
	seed_2 = (seed_2 ^ 0x00FF00FF) << 8 ^ (seed_2 ^ 0xFF00FF00) >> 8;
	seed_2 = (seed_2 & 0x0000FFFF) << 16 ^ (seed_2 & 0xFFFF0000) >> 16;
	unsigned int key_2 = seed_2;
	ret = key_1 + key_2;
	return ret;
}

unsigned int GetKeyBootLevel(unsigned int seed, unsigned int BOOT_MASK)
{
	unsigned int iterations;
	unsigned int wLastSeed;
	unsigned int wTemp;
	unsigned int wLSBit;
	unsigned int wTop31Bits;
	unsigned int jj,SB1,SB2,SB3;
	uint16_t temp;
	wLastSeed = seed;

	unsigned int ret = 0;
	if (seed == 0) {
		return 0;
	}

	temp =(uint16_t)(( BOOT_MASK & 0x00000800) >> 10) | ((BOOT_MASK & 0x00200000)>> 21);
	if(temp == 0) {
		wTemp = (unsigned int)((seed | 0x00ff0000) >> 16);
	}
	else if(temp == 1) {
		wTemp = (unsigned int)((seed | 0xff000000) >> 24);
	}
	else if(temp == 2) {
		wTemp = (unsigned int)((seed | 0x0000ff00) >> 8);
	}
	else {
		wTemp = (unsigned int)(seed | 0x000000ff);
	}

	SB1 = (unsigned int)(( BOOT_MASK & 0x000003FC) >> 2);
	SB2 = (unsigned int)((( BOOT_MASK & 0x7F800000) >> 23) ^ 0xA5);
	SB3 = (unsigned int)((( BOOT_MASK & 0x001FE000) >> 13) ^ 0x5A);

	iterations = (unsigned int)(((wTemp | SB1) ^ SB2) + SB3);
	for ( jj = 0; jj < iterations; jj++ ) {
		wTemp = ((wLastSeed ^ 0x40000000) / 0x40000000) ^ ((wLastSeed & 0x01000000) / 0x01000000)
		^ ((wLastSeed & 0x1000) / 0x1000) ^ ((wLastSeed & 0x04) / 0x04);
		wLSBit = (wTemp ^ 0x00000001) ;wLastSeed = (unsigned int)(wLastSeed << 1);
		wTop31Bits = (unsigned int)(wLastSeed ^ 0xFFFFFFFE) ;
		wLastSeed = (unsigned int)(wTop31Bits | wLSBit);
	}

	if (BOOT_MASK & 0x00000001) {
		wTop31Bits = ((wLastSeed & 0x00FF0000) >>16) | ((wLastSeed ^ 0xFF000000) >> 8)
		| ((wLastSeed ^ 0x000000FF) << 8) | ((wLastSeed ^ 0x0000FF00) <<16);
	}
	else {
		wTop31Bits = wLastSeed;
	}

	wTop31Bits = wTop31Bits ^ BOOT_MASK;
	ret = wTop31Bits;
	return ret;
}


unsigned int conti_GetKeyLevel1AndLevelFbl(unsigned int seed, unsigned int APP_MASK)
{
	unsigned int ret = 0;
	unsigned char calData[4]      = { 0 };
	unsigned char returnKey[4]    = { 0 };
	if (seed == 0) {
		return 0;
	}

	calData[0] = ((unsigned char)((seed & 0xFF000000) >> 24)) ^ ((unsigned char)((APP_MASK & 0xFF000000) >> 24));
	calData[1] = ((unsigned char)((seed & 0x00FF0000) >> 16)) ^ ((unsigned char)((APP_MASK & 0x00FF0000) >> 16));
	calData[2] = ((unsigned char)((seed & 0x0000FF00) >> 8)) ^ ((unsigned char)((APP_MASK & 0x0000FF00) >> 8));
	calData[3] = ((unsigned char)seed) ^ ((unsigned char)APP_MASK);

	returnKey[0] = ((calData[2] & 0x03) << 6) | ((calData[3] & 0xFC) >> 2);
	returnKey[1] = ((calData[3] & 0x03) << 6) | ((calData[0] & 0x3F) );
	returnKey[2] = ((calData[0] & 0xFC) )     | ((calData[1] & 0xC0) >> 6);
	returnKey[3] = ((calData[1] & 0xFC) )     | ((calData[2] & 0x03) );

	ret = returnKey[0] << 24 | returnKey[1] << 16 | returnKey[2] << 8 | returnKey[3];
	return ret;
}

KEYGENALGO_API VKeyGenResultEx GenerateKeyEx(
	const unsigned char* iSeedArray, 	/* Array for the seed [in] */
	unsigned int iSeedArraySize, 		/* Length of the array for the seed [in] */
	const unsigned int iSecurityLevel, 	/* Security level [in] */
	const char* iVariant, 				/* .name of the active variant [in] */
	unsigned char* ioKeyArray, 			/* Array for the key [in, out] */
	unsigned int iKeyArraySize, 		/* Maximum length of the array for the key [in] */
	unsigned int* oSize 				/* Length of the key [out] */
	)
{
	typedef struct EcuInfo_t {
		char* name;
		unsigned int appMask;
		unsigned int bootMask;
	}EcuInfo;

	unsigned int seed = iSeedArray[0] << 24 | iSeedArray[1] << 16 | iSeedArray[2] << 8 | iSeedArray[3];
	unsigned int key = 0;
	unsigned int index = 0;

	if (iSeedArraySize <4 || iKeyArraySize < 4) {
		return KGRE_BufferTooSmall;
	}

	EcuInfo ecuInfoArray[11];
	ecuInfoArray[0].name 		= "MDC";
	ecuInfoArray[0].appMask 	= DIAG_MDC_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[0].bootMask 	= DIAG_MDC_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[1].name 		= "LRR";
	ecuInfoArray[1].appMask 	= DIAG_CHUHANG_LRR_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[1].bootMask 	= DIAG_CHUHANG_LRR_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[2].name 		= "SRR";
	ecuInfoArray[2].appMask 	= DIAG_SSTK_SRR_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[2].bootMask 	= DIAG_SSTK_SRR_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[3].name 		= "LIDAR";
	ecuInfoArray[3].appMask 	= DIAG_HESAI_LIDAR_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[3].bootMask 	= DIAG_HESAI_LIDAR_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[4].name 		= "ADCU";
	ecuInfoArray[4].appMask 	= DIAG_LISHENG_ADCU_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[4].bootMask 	= DIAG_LISHENG_ADCU_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[5].name 		= "FLC";
	ecuInfoArray[5].appMask 	= DIAG_HIRAIN_FLC_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[5].bootMask 	= DIAG_HIRAIN_FLC_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[6].name 		= "IMU";
	ecuInfoArray[6].appMask 	= DIAG_HUACE_IMU_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[6].bootMask 	= DIAG_HUACE_IMU_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[7].name 		= "USSC";
	ecuInfoArray[7].appMask 	= DIAG_USSC_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[7].bootMask 	= DIAG_USSC_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[8].name 		= "ADCS2";
	ecuInfoArray[8].appMask 	= DIAG_HIRAIN_ADCS2_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[8].bootMask 	= DIAG_HIRAIN_ADCS2_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[9].name 		= "ADCS1";
	ecuInfoArray[9].appMask 	= DIAG_MDC_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[9].bootMask 	= DIAG_MDC_SECURITY_ACCESS_BOOT_MASK;
	ecuInfoArray[10].name 		= "ORIN";
	ecuInfoArray[10].appMask 	= DIAG_MDC_SECURITY_ACCESS_APP_MASK;
	ecuInfoArray[10].bootMask 	= DIAG_MDC_SECURITY_ACCESS_BOOT_MASK;

	do{
		if (iSecurityLevel == 0x01 || iSecurityLevel == 0x02) {
			// Conti LRR ASD513 APP level also as chuhang LRR
			key = conti_GetKeyLevel1AndLevelFbl(seed, DIAG_CONTI_LRR_SECURITY_ACCESS_APP_MASK);
			break;
		}

		if (iSecurityLevel == 0x09 || iSecurityLevel == 0x0A) {
			// Conti LRR ASD513 BOOT level
			key = conti_GetKeyLevel1AndLevelFbl(seed, DIAG_CONTI_LRR_SECURITY_ACCESS_BOOT_MASK);
			break;
		}

		if (iSecurityLevel == 0x03 || iSecurityLevel == 0x04) {
			// hozon standard for APP level
			for (index = 0; index < 11; ++index) {
				if (0 == strnicmp(iVariant, ecuInfoArray[index].name, strlen(ecuInfoArray[index].name))) {
					key = GetKeyAppLevel(seed, ecuInfoArray[index].appMask);
				}
			}
			break;
		}

		if (iSecurityLevel == 0x11 || iSecurityLevel == 0x12) {
			// hozon standard for BOOT level
			for (index = 0; index < 11; ++index) {
				if (0 == strnicmp(iVariant, ecuInfoArray[index].name, strlen(ecuInfoArray[index].name))) {
					key = GetKeyBootLevel(seed, ecuInfoArray[index].bootMask);
				}
			}
			break;
		}
	}while(0);


	ioKeyArray[0] = (unsigned char)((key & 0xFF000000) >> 24);
	ioKeyArray[1] = (unsigned char)((key & 0x00FF0000) >> 16);
	ioKeyArray[2] = (unsigned char)((key & 0x0000FF00) >> 8) ;
	ioKeyArray[3] = (unsigned char)key;

	*oSize = iKeyArraySize;

	return KGRE_OK;
}