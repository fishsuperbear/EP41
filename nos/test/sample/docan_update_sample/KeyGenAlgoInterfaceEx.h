
#ifndef KEY_GENERATE_H_
#define KEY_GENERATE_H_


#ifdef KEYGENALGO_EXPORTS
#define KEYGENALGO_API extern "C"  __declspec(dllexport)
#else
#define KEYGENALGO_API  __declspec(dllimport)
#endif


typedef enum  VKeyGenResult{
	KGRE_OK = 0,
	KGRE_BufferTooSmall = 1,
	KGRE_SecurityLevelInvalid = 2,
	KGRE_VariantInvalid = 3,
	KGRE_UnspecifiedError = 4,
}VKeyGenResultEx;

KEYGENALGO_API VKeyGenResultEx GenerateKeyEx(
	const unsigned char* iSeedArray, 	/* Array for the seed [in] */
	unsigned int iSeedArraySize, 		/* Length of the array for the seed [in] */
	const unsigned int iSecurityLevel, 	/* Security level [in] */
	const char* iVariant, 				/* Name of the active variant [in] */
	unsigned char* ioKeyArray, 			/* Array for the key [in, out] */
	unsigned int iKeyArraySize, 		/* Maximum length of the array for the key [in] */
	unsigned int* oSize                 /* Length of the key [out] */
	);


#endif