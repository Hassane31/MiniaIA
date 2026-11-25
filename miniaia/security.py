"""Command security policy.

Design rules
------------
* 100% independent of the LLM: pure functions, no network, no model. Whatever the
  model says, this module decides what may run.
* Commands are parsed here (NOT by a shell). Only plain commands, quoting and the
  pipe `|` are supported. Anything else (; & && || > < $VAR $(...) `...` newlines)
  is rejected, so there is no hidden second command and no injection through
  expansion.
* Three outcomes:
    SAFE      -> known read-only command, may run automatically
    APPROVAL  -> the human must approve (privileged, destructive, unknown, ...)
    BLOCKED   -> never runs, even if the human says yes
* Default is fail-closed: a command that is not on the read-only allow-list
  requires approval.
"""
from __future__ import annotations

import glob
import ipaddress
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

MAX_COMMAND_LENGTH = 2000


class Level(str, Enum):
    SAFE = "safe"
    APPROVAL = "approval"
    BLOCKED = "blocked"


# Categories that need a typed "yes" instead of a simple "y".
STRONG_CATEGORIES = frozenset({"destructive", "sensitive", "security-control", "user-management"})


@dataclass(frozen=True)
class Assessment:
    level: Level
    categories: tuple = ()
    reasons: tuple = ()
    segments: tuple = ()  # parsed argv of each pipeline stage

    @property
    def blocked(self) -> bool:
        return self.level is Level.BLOCKED

    @property
    def needs_approval(self) -> bool:
        return self.level is Level.APPROVAL

    @property
    def safe(self) -> bool:
        return self.level is Level.SAFE

    @property
    def strong_confirmation(self) -> bool:
        return bool(STRONG_CATEGORIES & set(self.categories))

    @property
    def summary(self) -> str:
        return "; ".join(self.reasons)


# --------------------------------------------------------------------------- authorized targets
TRUSTED_PROBE_IPS = frozenset({"1.1.1.1", "1.0.0.1", "8.8.8.8", "8.8.4.4", "9.9.9.9", "149.112.112.112"})
TRUSTED_PROBE_NAMES = frozenset({"localhost", "one.one.one.one", "dns.google", "dns.quad9.net"})


class TargetPolicy:
    """Which hosts may be probed (ping/traceroute) WITHOUT asking: this machine, the
    networks it is attached to, well-known public DNS probes, and what the user
    explicitly authorized in config (`authorized_targets`). Everything else asks."""

    def __init__(self, networks=(), hosts=()):
        self.networks = tuple(networks)
        self.hosts = frozenset(h.strip().lower() for h in hosts if h.strip())

    @classmethod
    def from_spec(cls, spec: str, networks=(), hosts=()) -> "TargetPolicy":
        nets, names = list(networks), list(hosts)
        for item in (x.strip() for x in spec.split(",")):
            if not item:
                continue
            try:
                nets.append(ipaddress.ip_network(item, strict=False))
            except ValueError:
                names.append(item)
        return cls(nets, names)

    def allows(self, host: str) -> bool:
        h = host.strip().lower().strip("[]").split("%", 1)[0]
        try:
            ip = ipaddress.ip_address(h)
        except ValueError:
            return h in TRUSTED_PROBE_NAMES or h in self.hosts
        if ip.is_loopback or str(ip) in TRUSTED_PROBE_IPS:
            return True
        return any(ip.version == n.version and ip in n for n in self.networks)


_DEFAULT_TARGETS = TargetPolicy()
_PROBE_COMMANDS = frozenset({"ping", "ping6", "traceroute", "tracepath"})
_PROBE_VALUE_OPTS = frozenset({"-c", "-i", "-W", "-w", "-s", "-t", "-I", "-S", "-Q", "-p", "-M", "-l", "-F",
                               "-m", "-q", "-g", "-z", "-f"})


def _probe_targets(args: list) -> list:
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a.startswith("-"):
            skip = a in _PROBE_VALUE_OPTS and a != "-f"
        else:
            out.append(a)
    return out


# --------------------------------------------------------------------------- parsing
class ParseError(ValueError):
    """The command uses syntax miniaIA deliberately does not support."""


_FORBIDDEN_UNQUOTED = {
    ";": "';' (command chaining)",
    "&": "'&' (background / chaining)",
    "<": "'<' (input redirection)",
    ">": "'>' (output redirection)",
    "`": "backticks (command substitution)",
    "(": "'(' (subshell)",
    ")": "')' (subshell)",
    "\n": "a newline",
    "\r": "a newline",
}


def _expand_glob(word: str, cwd: Optional[str], limit: int) -> list:
    base = cwd or os.getcwd()
    absolute = os.path.isabs(word)
    matches = sorted(glob.glob(word if absolute else os.path.join(base, word)))
    if not matches:
        return [word]
    if len(matches) > limit:
        raise ParseError(f"wildcard matches too many files (> {limit})")
    return matches if absolute else [os.path.relpath(m, base) for m in matches]


def parse(command: str, cwd: Optional[str] = None, glob_limit: int = 500) -> list:
    """Split a command into pipeline stages, each a list of argv strings."""
    if not command or not command.strip():
        raise ParseError("empty command")
    if len(command) > MAX_COMMAND_LENGTH:
        raise ParseError("command too long")

    segments: list = []
    argv: list = []
    buf: list = []
    in_tok = glob_ok = tilde = False
    quote: Optional[str] = None

    def end_token() -> None:
        nonlocal buf, in_tok, glob_ok, tilde
        if not in_tok:
            return
        word = "".join(buf)
        if tilde:
            word = os.path.expanduser(word)
        if glob_ok:
            argv.extend(_expand_glob(word, cwd, glob_limit))
        else:
            argv.append(word)
        buf, in_tok, glob_ok, tilde = [], False, False, False

    n = len(command)
    i = 0
    while i < n:
        c = command[i]
        if quote == "'":
            if c == "'":
                quote = None
            else:
                buf.append(c)
        elif quote == '"':
            if c == '"':
                quote = None
            elif c == "\\" and i + 1 < n and command[i + 1] in '"\\$`':
                buf.append(command[i + 1])
                i += 1
            elif c == "`" or (c == "$" and i + 1 < n and command[i + 1] in "({"):
                raise ParseError("command substitution is not supported")
            else:
                buf.append(c)
        else:
            if c in "'\"":
                quote = c
                in_tok = True
            elif c == "\\":
                if i + 1 >= n:
                    raise ParseError("trailing backslash")
                buf.append(command[i + 1])
                in_tok = True
                i += 1
            elif c in _FORBIDDEN_UNQUOTED:
                raise ParseError(f"unsupported shell syntax: {_FORBIDDEN_UNQUOTED[c]}")
            elif c == "$":
                raise ParseError("variable expansion ($) is not supported; use real paths")
            elif c in " \t":
                end_token()
            elif c == "|":
                if i + 1 < n and command[i + 1] == "|":
                    raise ParseError("unsupported shell syntax: '||'")
                end_token()
                if not argv:
                    raise ParseError("empty pipeline stage")
                segments.append(argv)
                argv = []
            else:
                if c == "~" and not in_tok:
                    tilde = True
                if c in "*?[":
                    glob_ok = True
                buf.append(c)
                in_tok = True
        i += 1

    if quote:
        raise ParseError("unterminated quote")
    end_token()
    if not argv:
        raise ParseError("empty pipeline stage" if segments else "empty command")
    segments.append(argv)
    return segments


# --------------------------------------------------------------------------- knowledge
TRUSTED_DIRS = frozenset({"/bin", "/sbin", "/usr/bin", "/usr/sbin", "/usr/local/bin", "/usr/local/sbin"})

SAFE_COMMANDS = frozenset({
    # files & text (read-only)
    "ls", "cat", "head", "tail", "grep", "egrep", "fgrep", "wc", "sort", "uniq", "cut", "tr",
    "stat", "file", "pwd", "echo", "tree", "find", "readlink", "realpath", "basename", "dirname",
    "du", "df", "cd", "which", "whereis", "md5sum", "sha256sum",
    # system info
    "date", "whoami", "id", "groups", "uname", "hostname", "uptime", "free", "nproc", "lscpu",
    "lsblk", "lsusb", "lspci", "lsmod", "lshw", "ps", "pgrep", "lsof", "who", "w", "last",
    "sensors", "acpi", "upower", "dmesg", "journalctl", "systemctl", "apt", "apt-cache", "dpkg",
    # network (read-only)
    "ip", "ss", "netstat", "ping", "traceroute", "tracepath", "dig", "nslookup", "host", "nmcli",
})

DESTRUCTIVE = frozenset({
    "rm", "rmdir", "mv", "dd", "shred", "truncate", "unlink", "fdisk", "sfdisk", "cfdisk", "parted",
    "gdisk", "sgdisk", "kill", "killall", "pkill", "skill", "chmod", "chown", "chgrp", "chattr",
    "reboot", "shutdown", "poweroff", "halt", "init", "telinit", "userdel", "groupdel", "deluser",
    "delgroup", "passwd", "usermod", "chpasswd", "umount", "swapoff", "crontab", "losetup",
    "cryptsetup", "lvremove", "vgremove", "pvremove", "mdadm",
})
MODIFY_FILES = frozenset({"touch", "mkdir", "cp", "ln", "tee", "install"})
WRAPPERS = frozenset({
    "xargs", "env", "nohup", "nice", "ionice", "timeout", "watch", "setsid", "stdbuf", "time",
    "chroot", "unshare", "nsenter", "strace", "ltrace", "gdb", "script", "screen", "tmux",
})
SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish", "csh", "tcsh"})
INTERPRETERS = frozenset({
    "python", "python2", "python3", "perl", "ruby", "node", "nodejs", "php", "lua",
    "awk", "gawk", "mawk", "expect", "tclsh",
})
SENSITIVE_COMMANDS = frozenset({"env", "printenv", "secret-tool", "pass", "unshadow"})
SECURITY_COMMANDS = frozenset({"setenforce", "aa-disable", "aa-teardown", "aa-complain", "auditctl", "setsebool"})
NETWORK_ACTIVE = frozenset({
    "nmap", "masscan", "hping3", "arp-scan", "netdiscover", "nikto", "tcpdump", "tshark", "ettercap",
})
SECURITY_SERVICES = frozenset({
    "ufw", "firewalld", "apparmor", "auditd", "fail2ban", "nftables", "iptables", "netfilter-persistent",
    "clamav-daemon", "clamav-freshclam", "selinux",
})
AWKS = frozenset({"awk", "gawk", "mawk", "nawk"})
# Only trivially safe awk programs run without approval: print fields / count lines, optionally
# restricted to one line number. No system(), getline, pipes, redirection, variables or functions.
_AWK_FIELD = r"(?:\$(?:\d+|NF))"
_AWK_SAFE_PROGRAM = re.compile(
    r"^\s*(?:NR\s*(?:==|>=|<=|>|<)\s*\d+\s*)?\{\s*print\s*(?:" + _AWK_FIELD + r"(?:\s*,\s*" + _AWK_FIELD + r")*)?\s*\}\s*$"
    r"|^\s*END\s*\{\s*print\s+NR\s*\}\s*$")
_AWK_SEPARATOR = re.compile(r"^(?:[:,;| ]|\\t)$")

USER_MGMT = frozenset({"useradd", "adduser", "groupadd", "addgroup", "usermod", "userdel", "deluser", "delgroup",
                       "groupdel", "groupmod", "passwd", "chage", "gpasswd", "vipw"})
PRIVILEGED_GROUPS = frozenset({"sudo", "wheel", "admin", "root", "adm", "shadow", "disk"})
PASSWORD_PIPE_COMMANDS = frozenset({"chpasswd", "newusers"})
PRIV_WRAPPERS = frozenset({"sudo", "doas", "pkexec"})
_SUDO_VALUE_OPTS = frozenset({"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-T", "-U", "-R", "-c"})

_SENSITIVE_PATTERNS = [re.compile(p) for p in (
    r"(^|/)etc/(shadow|gshadow|sudoers|security/opasswd)",
    r"(^|/)etc/ssh/ssh_host_[a-z0-9_]*_key$",
    r"(^|/)\.ssh(/|$)",
    r"(^|/)\.gnupg(/|$)",
    r"(^|/)\.aws(/|$)",
    r"(^|/)\.netrc$",
    r"(^|/)\.pgpass$",
    r"(^|/)\.(bash|zsh)_history$",
    r"(^|/)(logins\.json|key4\.db|cookies\.sqlite)$",
    r"^/proc/[^/]+/environ$",
    r"(^|/)id_(rsa|dsa|ecdsa|ed25519)",
    r"\.(pem|key|p12|pfx|kdbx)$",
    r"(^|/)etc/NetworkManager/system-connections",
    r"(^|/)etc/wpa_supplicant",
)]

_PROTECTED = {
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib32", "/lib64", "/opt", "/proc",
    "/root", "/run", "/sbin", "/srv", "/sys", "/usr", "/var",
}


# --------------------------------------------------------------------------- helpers
def _norm(path: str, cwd: Optional[str]) -> str:
    return os.path.normpath(os.path.join(cwd or os.getcwd(), os.path.expanduser(path)))


def _protected_paths() -> set:
    home = os.path.expanduser("~")
    paths = set(_PROTECTED) | {os.path.normpath(home)}
    return paths | {os.path.realpath(p) for p in paths}


def _is_protected(path: str, cwd: Optional[str]) -> bool:
    prot = _protected_paths()
    n = _norm(path, cwd)
    return n in prot or os.path.realpath(n) in prot


def _split_args(args: list) -> tuple:
    opts, targets, rest = [], [], False
    for a in args:
        if rest:
            targets.append(a)
        elif a == "--":
            rest = True
        elif a.startswith("-") and len(a) > 1:
            opts.append(a)
        else:
            targets.append(a)
    return opts, targets


def _short_flags(args: list) -> set:
    chars: set = set()
    for a in args:
        if a.startswith("-") and not a.startswith("--"):
            chars.update(a[1:])
    return chars


def _recursive(opts: list) -> bool:
    return "--recursive" in opts or bool({"r", "R"} & _short_flags(opts))


def _is_sensitive(arg: str, cwd: Optional[str]) -> bool:
    candidates = {arg}
    if "=" in arg:
        candidates.add(arg.split("=", 1)[1])
    for cand in list(candidates):
        if not cand:
            continue
        candidates.add(_norm(cand, cwd))
    for cand in candidates:
        if cand.endswith(".pub"):
            continue
        if any(rx.search(cand) for rx in _SENSITIVE_PATTERNS):
            return True
    return False


def _safe_awk(args: list) -> bool:
    """True only for: awk [-F<sep>] '<trivial print program>' [files]."""
    i, sep_ok = 0, True
    if args and args[0] == "-F" and len(args) > 1:
        sep_ok, i = bool(_AWK_SEPARATOR.match(args[1])), 2
    elif args and args[0].startswith("-F") and len(args[0]) > 2:
        sep_ok, i = bool(_AWK_SEPARATOR.match(args[0][2:])), 1
    if not sep_ok or i >= len(args) or args[i].startswith("-"):
        return False
    if not _AWK_SAFE_PROGRAM.match(args[i]):
        return False
    return not any(a.startswith("-") for a in args[i + 1:])


def _service(name: str) -> str:
    name = name.lower()
    return name[:-8] if name.endswith(".service") else name


Flags = list  # list of (category, reason)


# --------------------------------------------------------------------------- per-command guards
def _g_find(args: list) -> Flags:
    bad = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0", "-fprintf", "-fls"}
    return [("destructive", f"find {a} can modify files or run commands") for a in args if a in bad]


_SYSTEMCTL_READ = {
    "status", "is-active", "is-enabled", "is-failed", "list-units", "list-unit-files", "list-timers",
    "list-sockets", "list-dependencies", "show", "cat", "list-jobs", "get-default",
}
_POWER_VERBS = {"poweroff", "reboot", "halt", "kexec", "suspend", "hibernate", "rescue", "emergency"}


def _g_systemctl(args: list) -> Flags:
    _, targets = _split_args(args)
    if not targets or targets[0] in _SYSTEMCTL_READ:
        return []
    verb = targets[0]
    flags: Flags = [("modifies-system", f"systemctl {verb} changes system state")]
    if verb in _POWER_VERBS:
        flags.append(("destructive", f"systemctl {verb} changes the machine power state"))
    if verb in {"stop", "disable", "mask", "kill"} and any(_service(t) in SECURITY_SERVICES for t in targets[1:]):
        flags.append(("security-control", "this would disable a security service"))
    return flags


def _g_service(args: list) -> Flags:
    _, targets = _split_args(args)
    if len(targets) >= 2 and targets[1] == "status":
        return []
    flags: Flags = [("modifies-system", "service changes system state")]
    if len(targets) >= 2 and targets[1] in {"stop", "disable"} and _service(targets[0]) in SECURITY_SERVICES:
        flags.append(("security-control", "this would disable a security service"))
    return flags


def _g_journalctl(args: list) -> Flags:
    bad = ("--vacuum", "--rotate", "--flush", "--sync", "--relinquish", "--smart-relinquish", "--setup-keys", "--update-catalog")
    return [("modifies-system", "journalctl option changes the journal")] if any(a.startswith(bad) for a in args) else []


def _g_dmesg(args: list) -> Flags:
    long_bad = {"--clear", "--read-clear", "--console-off", "--console-on", "--console-level"}
    if long_bad & set(args) or {"c", "C", "D", "E", "n"} & _short_flags(args):
        return [("modifies-system", "dmesg option clears the log or changes console settings")]
    return []


_IP_MUTATORS = {"add", "del", "delete", "set", "flush", "replace", "change", "append", "prepend", "save", "restore", "exec"}


def _g_ip(args: list) -> Flags:
    return [("modifies-system", "ip command changes network configuration")] if _IP_MUTATORS & set(args) else []


def _g_ss(args: list) -> Flags:
    if "--kill" in args or "K" in _short_flags(args):
        return [("modifies-system", "ss -K kills sockets")]
    return []


_NMCLI_MUTATORS = {"up", "down", "add", "modify", "delete", "connect", "disconnect", "reload", "edit", "clone",
                   "import", "load", "save", "on", "off", "hotspot"}


def _g_nmcli(args: list) -> Flags:
    flags: Flags = []
    if _NMCLI_MUTATORS & set(args):
        flags.append(("modifies-system", "nmcli changes network configuration"))
    if "--show-secrets" in args or "s" in _short_flags(args):
        flags.append(("sensitive", "nmcli can print saved Wi-Fi passwords"))
    return flags


_APT_READ = {"list", "show", "search", "policy", "depends", "rdepends", "showpkg", "madison", "help", "changelog"}
_APT_REMOVE = {"remove", "purge", "autoremove", "autopurge", "dist-upgrade", "full-upgrade"}


def _g_apt(args: list) -> Flags:
    _, targets = _split_args(args)
    if not targets or targets[0] in _APT_READ:
        return []
    if targets[0] in _APT_REMOVE:
        return [("destructive", f"apt {targets[0]} removes or replaces software")]
    return [("modifies-system", f"apt {targets[0]} changes installed software")]


_DPKG_READ = {"-l", "--list", "-L", "--listfiles", "-s", "--status", "-S", "--search", "-p", "--print-avail",
              "--version", "-C", "--audit", "-V", "--verify"}


def _g_dpkg(args: list) -> Flags:
    if any(a in {"-r", "--remove", "-P", "--purge"} for a in args):
        return [("destructive", "dpkg removes software")]
    if not args or any(a in _DPKG_READ for a in args):
        return []
    return [("modifies-system", "dpkg changes installed software")]


def _g_sort(args: list) -> Flags:
    if "o" in _short_flags(args) or any(a.startswith("--output") for a in args):
        return [("modifies-files", "sort -o writes to a file")]
    return []


def _g_uniq(args: list) -> Flags:
    _, targets = _split_args(args)
    return [("modifies-files", "uniq with two file arguments writes the second one")] if len(targets) > 1 else []


def _g_date(args: list) -> Flags:
    if "s" in _short_flags(args) or "--set" in args or any(a.startswith("--set=") for a in args):
        return [("modifies-system", "date -s changes the system clock")]
    return []


def _g_hostname(args: list) -> Flags:
    _, targets = _split_args(args)
    return [("modifies-system", "hostname NAME changes the hostname")] if targets else []


def _g_ufw(args: list) -> Flags:
    _, targets = _split_args(args)
    if targets and targets[0] in {"status", "version", "show"}:
        return []
    return [("security-control", "this changes firewall configuration")]


_FW_BAD = {"-F", "--flush", "-X", "--delete-chain", "-P", "--policy", "flush", "delete", "destroy"}


def _g_firewall(args: list) -> Flags:
    if _FW_BAD & set(args):
        return [("security-control", "this flushes or alters firewall rules")]
    return []


_GUARDS: dict = {
    "find": _g_find, "systemctl": _g_systemctl, "service": _g_service, "journalctl": _g_journalctl,
    "dmesg": _g_dmesg, "ip": _g_ip, "ss": _g_ss, "nmcli": _g_nmcli, "apt": _g_apt, "apt-get": _g_apt,
    "dpkg": _g_dpkg, "sort": _g_sort, "uniq": _g_uniq, "date": _g_date, "hostname": _g_hostname,
    "ufw": _g_ufw, "iptables": _g_firewall, "ip6tables": _g_firewall, "nft": _g_firewall,
    "nftables": _g_firewall, "ebtables": _g_firewall, "arptables": _g_firewall,
}


# --------------------------------------------------------------------------- blocking & privilege
def _strip_privilege(argv: list) -> tuple:
    """Return (privileged, inner_argv, blocked_reason)."""
    privileged = False
    cur = list(argv)
    while cur and os.path.basename(cur[0]) in PRIV_WRAPPERS:
        privileged = True
        i, shell_flag = 1, False
        while i < len(cur) and cur[i].startswith("-") and cur[i] != "-":
            opt = cur[i]
            if opt == "--":
                i += 1
                break
            if opt in _SUDO_VALUE_OPTS:
                i += 2
                continue
            if opt in ("--login", "--shell") or (not opt.startswith("--") and {"i", "s"} & set(opt[1:])):
                shell_flag = True
            i += 1
        cur = cur[i:]
        if shell_flag or not cur:
            return privileged, [], "an interactive root shell is not allowed; use 'sudo <single command>'"
    return privileged, cur, None


def _user_mgmt_block(name: str, args: list) -> Optional[str]:
    if name in PASSWORD_PIPE_COMMANDS:
        return f"'{name}' takes passwords from input; passwords must never pass through miniaIA. Run 'sudo passwd <user>' yourself"
    if name in ("useradd", "usermod", "adduser"):
        for i, a in enumerate(args):
            if a in ("-p", "--password") or a.startswith("--password="):
                return "passwords on a command line leak into process lists and logs; set it yourself with 'sudo passwd <user>'"
            if (a in ("-u", "--uid") and i + 1 < len(args) and args[i + 1] == "0") or a == "--uid=0":
                return "creating or changing an account to UID 0 would create a second root account"
    return None


def _blocked_reason(name: str, args: list, cwd: Optional[str]) -> Optional[str]:
    why = _user_mgmt_block(name, args)
    if why:
        return why
    if name in SHELLS and any(re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", a) for a in args):
        return f"'{name} -c' hides the real command from the safety checks"
    if name in {"eval", "exec", "source", "."}:
        return f"'{name}' runs arbitrary code that cannot be analysed"
    if name == "su":
        return "switching user with su is not allowed"
    if name.startswith("mkfs") or name in {"mke2fs", "mkswap", "wipefs"}:
        return "formatting/wiping filesystems is blocked; do it manually in a terminal"
    opts, targets = _split_args(args)
    if name == "dd" and any(a.startswith("of=/dev/") for a in args):
        return "writing directly to a device with dd is blocked; do it manually in a terminal"
    if name == "shred" and any(t.startswith("/dev/") for t in targets):
        return "shredding a device is blocked"
    if name == "rm":
        if "--no-preserve-root" in opts:
            return "--no-preserve-root is blocked"
        if _recursive(opts) and any(_is_protected(t, cwd) for t in targets):
            return "recursive removal of a system directory or your home directory is blocked"
    if name == "mv" and any(_is_protected(t, cwd) for t in targets[:-1]):
        return "moving a system directory or your home directory is blocked"
    if name in {"chmod", "chown", "chgrp"} and _recursive(opts) and any(_is_protected(t, cwd) for t in targets):
        return "recursive permission/ownership change on a system directory is blocked"
    return None


# --------------------------------------------------------------------------- assessment
def _assess_segment(argv: list, cwd: Optional[str], targets: Optional[TargetPolicy] = None) -> tuple:
    cats: set = set()
    reasons: list = []

    def flag(cat: str, why: str) -> None:
        cats.add(cat)
        reasons.append(why)

    privileged, inner, block = _strip_privilege(argv)
    if block:
        return cats, reasons, block
    if privileged:
        flag("privileged", "runs with elevated privileges (sudo)")

    exe, args = inner[0], inner[1:]
    if any(ch.isspace() for ch in exe):
        return cats, reasons, ("the program name contains spaces: the whole command was passed as ONE quoted word. "
                               "Write the command exactly as typed in a terminal, without quotes around all of it")
    name = os.path.basename(exe)
    trusted = "/" not in exe or os.path.dirname(exe) in TRUSTED_DIRS
    if not trusted:
        flag("unknown", f"runs a program by path ({exe}) instead of a trusted system command")

    reason = _blocked_reason(name, args, cwd)
    if reason:
        return cats, reasons, reason

    if name in DESTRUCTIVE:
        flag("destructive", f"'{name}' can delete data or change system state irreversibly")
    if name in MODIFY_FILES:
        flag("modifies-files", f"'{name}' creates or overwrites files")
    if name in WRAPPERS:
        flag("wrapper", f"'{name}' runs another command that cannot be fully analysed")
    awk_ok = name in AWKS and trusted and _safe_awk(args)
    if (name in SHELLS or name in INTERPRETERS) and not awk_ok:
        flag("interpreter", f"'{name}' can run arbitrary code")
    if name in USER_MGMT:
        flag("user-management", f"'{name}' creates, changes or removes user accounts or passwords")
        groups = {g for a in args for g in a.split(",")} & PRIVILEGED_GROUPS
        if name in ("usermod", "useradd", "adduser", "gpasswd") and groups:
            flag("security-control", f"grants membership of a privileged group ({', '.join(sorted(groups))})")
    if name in SENSITIVE_COMMANDS:
        flag("sensitive", f"'{name}' may expose passwords, tokens or other secrets")
    if name in SECURITY_COMMANDS:
        flag("security-control", f"'{name}' changes security protections")
    if name in NETWORK_ACTIVE:
        flag("network-active", f"'{name}' probes or captures network traffic; only use it on networks you own or may test")

    guard: Optional[Callable] = _GUARDS.get(name)
    if guard:
        for cat, why in guard(args):
            flag(cat, why)

    if name in _PROBE_COMMANDS:
        policy = targets or _DEFAULT_TARGETS
        outside = [t for t in _probe_targets(args) if not policy.allows(t)]
        if outside:
            flag("network-active", f"'{name}' would probe a host outside your authorized networks: {', '.join(outside)}")
        if name.startswith("ping") and "f" in _short_flags(args):
            flag("network-active", "ping -f floods the target")

    safe_candidate = trusted and (name in SAFE_COMMANDS or awk_ok)
    if not cats and not safe_candidate:
        flag("unknown", f"'{name}' is not on the read-only allow-list")

    for a in args:
        if _is_sensitive(a, cwd):
            flag("sensitive", f"touches a sensitive file or location ({a})")
            break
    return cats, reasons, None


def assess(command: str, cwd: Optional[str] = None, targets: Optional[TargetPolicy] = None) -> Assessment:
    try:
        segments = parse(command, cwd)
    except ParseError as exc:
        return Assessment(Level.BLOCKED, ("unsupported-syntax",), (str(exc),))
    seg_t = tuple(tuple(s) for s in segments)

    cats: set = set()
    reasons: list = []
    for seg in segments:
        c, r, blocked = _assess_segment(seg, cwd, targets)
        if blocked:
            return Assessment(Level.BLOCKED, ("blocked",), (blocked,), seg_t)
        cats |= c
        reasons.extend(r)
    reasons = list(dict.fromkeys(reasons))
    if not cats:
        return Assessment(Level.SAFE, (), (), seg_t)
    return Assessment(Level.APPROVAL, tuple(sorted(cats)), tuple(reasons), seg_t)


# --------------------------------------------------------------------------- file reading policy
READ_ALLOWED_ROOTS = ("/etc", "/var/log", "/tmp", "/usr/share/doc", "/proc/cpuinfo", "/proc/meminfo", "/proc/version",
                      "/proc/uptime", "/proc/loadavg", "/proc/mounts", "/proc/net", "/sys/class/power_supply",
                      "/sys/class/thermal")
# Files inside an allowed root may be symlinks (e.g. /etc/os-release -> /usr/lib/os-release); they may
# resolve into these locations only. Secrets and devices are still checked on the resolved path.
READ_SYMLINK_TARGETS = ("/usr/lib/os-release", "/usr/share", "/run/systemd/resolve", "/sys/devices", "/proc")
_READ_BLOCKED = [re.compile(p) for p in (r"^/dev(/|$)", r"^/proc/kcore", r"^/proc/(\d+|self)/(mem|pagemap|maps|smaps)",
                                         r"^/sys/kernel/debug", r"^/proc/kmsg")]


def resolve_path(path: str, cwd: Optional[str] = None) -> str:
    return os.path.realpath(_norm(path, cwd))


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _sensitive_path(p: str) -> bool:
    return not p.endswith(".pub") and any(rx.search(p) for rx in _SENSITIVE_PATTERNS)


def assess_read_path(path: str, cwd: Optional[str] = None) -> Assessment:
    """Policy for the read_file tool (same three outcomes as commands)."""
    if not path or "\x00" in path or "\n" in path:
        return Assessment(Level.BLOCKED, ("invalid-path",), ("invalid path",))
    shown, resolved = _norm(path, cwd), resolve_path(path, cwd)   # symlinks are followed before judging
    for cand in (shown, resolved):
        if any(rx.search(cand) for rx in _READ_BLOCKED):
            return Assessment(Level.BLOCKED, ("blocked",), (f"reading {cand} is never allowed (device or kernel memory)",))
    if _sensitive_path(shown) or _sensitive_path(resolved):
        return Assessment(Level.APPROVAL, ("sensitive",), (f"{resolved} may contain passwords, keys or other secrets",))
    home = os.path.realpath(os.path.expanduser("~"))
    if _under(resolved, home):
        rel = os.path.relpath(resolved, home).split(os.sep)
        if any(part.startswith(".") for part in rel if part not in (".", "")):
            return Assessment(Level.APPROVAL, ("outside-permitted-locations",),
                              ("hidden files/folders in your home often hold credentials or tokens",))
        return Assessment(Level.SAFE)
    if any(_under(shown, r) for r in READ_ALLOWED_ROOTS) and any(
            _under(resolved, r) for r in READ_ALLOWED_ROOTS + READ_SYMLINK_TARGETS):
        return Assessment(Level.SAFE)
    return Assessment(Level.APPROVAL, ("outside-permitted-locations",),
                      (f"{resolved} is outside the automatically readable locations (home, /etc, /var/log, /tmp)",))


# --------------------------------------------------------------------------- file ACTION policy
WRITE_BLOCKED_TREES = ("/boot", "/proc", "/sys", "/dev", "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64",
                       "/etc", "/var", "/root", "/run", "/srv", "/opt")
WRITE_FREE_ROOTS = ("/tmp", "/var/tmp", "/media", "/mnt")


def assess_write_path(path: str, cwd: Optional[str] = None, delete: bool = False) -> Assessment:
    """Policy for create_folder / create_file / delete_path. Creating asks (y/N); deleting needs a typed yes.
    System locations are never touched by these tools (use reviewed sudo commands for that)."""
    if not path or "\x00" in path or "\n" in path:
        return Assessment(Level.BLOCKED, ("invalid-path",), ("invalid path",))
    shown, resolved = _norm(path, cwd), resolve_path(path, cwd)
    for cand in (shown, resolved):
        if _is_protected(cand, None):
            return Assessment(Level.BLOCKED, ("blocked",), (f"{cand} is a system directory or your home directory itself",))
        if cand not in WRITE_FREE_ROOTS and any(_under(cand, t) for t in WRITE_BLOCKED_TREES if not any(_under(cand, f) for f in WRITE_FREE_ROOTS)):
            return Assessment(Level.BLOCKED, ("blocked",), (f"{cand} is a system location; these tools only work in your home folder, /tmp, /media or /mnt",))
    cats = {"destructive" if delete else "modifies-files"}
    reasons = ["moves it to the Trash (recoverable)" if delete else "creates a new item on disk"]
    if _sensitive_path(shown) or _sensitive_path(resolved):
        cats.add("sensitive")
        reasons.append("this location may hold secrets (keys, credentials)")
    home = os.path.realpath(os.path.expanduser("~"))
    if _under(resolved, home):
        if any(part.startswith(".") for part in os.path.relpath(resolved, home).split(os.sep) if part not in (".", "")):
            reasons.append("hidden file/folder in your home (often application settings)")
    elif not any(_under(resolved, r) for r in WRITE_FREE_ROOTS):
        cats.add("outside-permitted-locations")
        reasons.append(f"{resolved} is outside your home folder, /tmp, /media and /mnt")
    return Assessment(Level.APPROVAL, tuple(sorted(cats)), tuple(reasons))


# --------------------------------------------------------------------------- redaction
_REDACTIONS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED PRIVATE KEY]"),
    (re.compile(r"(?i)\b(authorization\s*:\s*(?:bearer|basic)\s+)[^\s'\"]+"), r"\1[REDACTED]"),
    (re.compile(r"(?<=://)([^/\s:@]+):([^/\s@]+)@"), r"\1:[REDACTED]@"),
    (re.compile(
        r"(?i)\b([\w.-]*(?:password|passwd|pwd|passphrase|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|credential)[\w.-]*)"
        r"(\s*[=:]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"), r"\1\2[REDACTED]"),
    (re.compile(r"(?i)(--?(?:password|passwd|pass|token|secret|api-key|apikey)[ =])(\S+)"), r"\1[REDACTED]"),
    (re.compile(
        r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"), "[REDACTED]"),
    (re.compile(r"\$(?:1|2[abxy]?|5|6|y|7)\$[A-Za-z0-9./$,=-]{8,}"), "[REDACTED HASH]"),
]


def redact(text: str) -> str:
    """Mask passwords, tokens, keys and password hashes (used for logs and LLM context)."""
    for rx, repl in _REDACTIONS:
        text = rx.sub(repl, text)
    return text
