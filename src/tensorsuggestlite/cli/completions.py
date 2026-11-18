"""Shell completion generators for the TensorSuggestLite CLI."""

import sys

PROG_NAME = "tensorsuggestlite"


def generate_completions(shell: str) -> int:
    """Generate shell autocompletion script.

    Args:
        shell: Target shell ('bash', 'zsh', 'fish', 'powershell').

    Returns:
        int: 0 on success, 2 on unsupported shell.
    """
    shells = {"bash", "zsh", "fish", "powershell"}
    if shell.lower() not in shells:
        sys.stderr.write(
            f"error: Unsupported shell '{shell}'. Supported: {', '.join(sorted(shells))}\n"
        )
        sys.stderr.flush()
        return 2

    sh = shell.lower()
    if sh == "bash":
        script = f"""# bash completion for {PROG_NAME}
_{PROG_NAME}_completion() {{
    local cur prev words cword
    _init_completion || return
    local commands="gui inspect train convert predict history completions version help"
    local global_opts="-V --version -q --quiet -v --verbose --json --no-color -h --help"

    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$global_opts" -- "$cur") )
        return 0
    fi

    case "$prev" in
        {PROG_NAME})
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
            return 0
            ;;
        completions)
            COMPREPLY=( $(compgen -W "bash zsh fish powershell" -- "$cur") )
            return 0
            ;;
        help)
            COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
            return 0
            ;;
    esac
}}
complete -F _{PROG_NAME}_completion {PROG_NAME}
"""
    elif sh == "zsh":
        script = f"""#compdef {PROG_NAME}
_arguments \\
  '(-V --version)'{{-V,--version}}'[Print version]' \\
  '(-q --quiet)'{{-q,--quiet}}'[Quiet mode]' \\
  '(-v --verbose)'{{-v,--verbose}}'[Verbose mode]' \\
  '--json[JSON output mode]' \\
  '--no-color[Disable color output]' \\
  '1:command:(gui inspect train convert predict history completions version help)' \\
  '*::arg:->args'
"""
    elif sh == "fish":
        script = f"""# fish completion for {PROG_NAME}
complete -c {PROG_NAME} -s V -l version -d 'Print version and exit'
complete -c {PROG_NAME} -s q -l quiet -d 'Suppress non-error diagnostics'
complete -c {PROG_NAME} -s v -l verbose -d 'Verbose diagnostics to stderr'
complete -c {PROG_NAME} -l json -d 'Machine-readable JSON output'
complete -c {PROG_NAME} -l no-color -d 'Disable ANSI colored output'
complete -c {PROG_NAME} -f -a 'gui inspect train convert predict history completions version help'
"""
    else:  # powershell
        script = f"""# powershell completion for {PROG_NAME}
Register-ArgumentCompleter -Native -CommandName '{PROG_NAME}' -ScriptBlock {{
    param($wordToComplete, $commandAst, $cursorPosition)
    $commands = @('gui', 'inspect', 'train', 'convert', 'predict', 'history', 'completions', 'version', 'help')
    $commands | Where-Object {{ $_ -like "$wordToComplete*" }} | ForEach-Object {{
        [System.Management.Automation.CompletionResult]::new($_, $_, 'ParameterValue', $_)
    }}
}}
"""
    sys.stdout.write(script)
    sys.stdout.flush()
    return 0
