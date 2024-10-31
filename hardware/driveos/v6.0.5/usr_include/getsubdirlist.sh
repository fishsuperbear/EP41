#!/bin/bash
SHDIR="$(dirname "$(realpath "$BASH_SOURCE")")"
TOTALPATH="./"
TOTALPATHLEN=${#TOTALPATH}
HZSCPFILELIST="filelist.txt"
BLACKLIST=('filelist' 'cuda')
DIRLIST="dirlist.txt"

PREFIX="../../../../driveos/v6.0.5/usr_include/"

:> ${HZSCPFILELIST}
find ${TOTALPATH} -regextype posix-extended -regex ".*" > ${HZSCPFILELIST}
for BLACKI in ${BLACKLIST[@]}
do
	sed -i "/${BLACKI}/d" ${HZSCPFILELIST}
done

FILEFULL=""

:> ${DIRLIST}

while read FILEI
do
    if [ -f ${FILEI} ]; then
        continue
    fi
    FILECUT=`echo ${FILEI} | cut -c$((TOTALPATHLEN+1))-`
    echo -n "${PREFIX}${FILECUT};" >> ${DIRLIST}
    #FILEFULL="${PREFIX}${FILECUT};${FILEFULL}"
done <${HZSCPFILELIST}

rm -f ${HZSCPFILELIST}

#${FILEFULL} > ${DIRLIST}
