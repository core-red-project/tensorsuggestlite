"""Generates a roff-formatted man page for TensorSuggestLite.

Used for packaging in Debian/Ubuntu, Homebrew formulas, and offline Unix systems.
"""

from __future__ import annotations

from datetime import date

from tensorsuggestlite import __version__


def generate_man_page() -> str:
    """Generate roff man page content.

    Returns:
        str: Formatted roff source code.
    """
    today_str = date.today().isoformat()
    return f""".TH TENSORSUGGESTLITE 1 "{today_str}" "{__version__}" "TensorSuggestLite Manual"
.SH NAME
tensorsuggestlite \\- Configuration-driven text classification and TFLite model generation
.SH SYNOPSIS
.B tensorsuggestlite
[\\fIOPTIONS\\fR] [\\fICOMMAND\\fR]
.SH DESCRIPTION
.B TensorSuggestLite
is an enterprise-grade text classification toolkit powered by Keras
and TensorFlow Lite. It parses dataset definitions from JSON, YAML, or TOML,
performs deep dataset profiling, trains regularized neural classifiers, and exports
quantized .tflite binaries optimized for edge environments.
.SH OPTIONS
.TP
\\fB\\-V\\fR, \\fB\\-\\-version\\fR
Print version information and exit.
.TP
\\fB\\-q\\fR, \\fB\\-\\-quiet\\fR
Suppress diagnostic and progress messages on stderr.
.TP
\\fB\\-v\\fR, \\fB\\-\\-verbose\\fR
Emit granular debug diagnostics to stderr.
.TP
\\fB\\-\\-json\\fR
Emit machine-readable JSON data to stdout.
.TP
\\fB\\-\\-no\\-color\\fR
Disable ANSI colors in terminal outputs.
.SH COMMANDS
.TP
\\fBgui\\fR
Launch the interactive Dear PyGui desktop interface.
.TP
\\fBinspect\\fR \\fIFILE\\fR
Inspect and validate a dataset configuration file with dataset profiling.
.TP
\\fBtrain\\fR \\fIFILE\\fR [\\fIOPTIONS\\fR]
Train a neural network classifier and serialize artifacts.
.TP
\\fBconvert\\fR \\fIMODEL_DIR\\fR [\\fIOPTIONS\\fR]
Convert trained Keras model into optimized TFLite binary.
.TP
\\fBpredict\\fR \\fIMODEL_DIR\\fR [\\fITEXT\\fR]
Classify input text using Keras or TFLite runtime.
.TP
\\fBhistory\\fR \\fIMODEL_DIR\\fR
Inspect metadata and historical performance logs.
.TP
\\fBcompletions\\fR \\fISHELL\\fR
Emit shell completion script (bash, zsh, fish, powershell).
.TP
\\fBhelp\\fR [\\fICOMMAND\\fR]
Display help for tensorsuggestlite or a specific command.
.SH AUTHORS
Core Red Project.
.SH COPYRIGHT
MIT License.
"""


__all__ = ["generate_man_page"]
