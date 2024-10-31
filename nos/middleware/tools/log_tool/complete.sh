# 设置 word break characters，将默认的空格和 tab 加上冒号
COMP_WORDBREAKS=${COMP_WORDBREAKS//:/}

_hz_log_tools_()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="continue setloglevel convert"
    opts_appid_ctx="IGNORE.IGNORE:kCritical IGNORE.IGNORE:kError IGNORE.IGNORE:kWarn IGNORE.IGNORE:kInfo IGNORE.IGNORE:kDebug IGNORE.IGNORE:kTrace"

    # Complete the options
    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    elif [ $COMP_CWORD -eq 2 ]; then
        if [[ ${prev} == "setloglevel" ]]; then
            COMPREPLY=( $(compgen -W "${opts_appid_ctx}" -- ${cur}) )
        else
            # Default file and directory completion
            COMPREPLY=( $(compgen -f -- ${cur}) )
        fi
    else
        # Default file and directory completion
        COMPREPLY=( $(compgen -f -- ${cur}) )
    fi
}

complete -F _hz_log_tools_ -o filenames -o dirnames log