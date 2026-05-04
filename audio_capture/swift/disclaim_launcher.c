// Disclaim-spawn launcher for the bundled `capture_audio` Swift helper.
//
// macOS TCC normally inherits the "responsible process" from a binary's
// parent: when Python's subprocess.Popen forks/execs `capture_audio`, TCC
// attributes the screen-recording / microphone requests to whichever
// terminal launched Python (Terminal.app, iTerm, Ghostty, …), not to the
// bundled helper. As a result, the helper's bundle identity is never
// added to System Settings → Privacy & Security and the user can't grant
// (or revoke) permissions per-app for Chirp.
//
// `responsibility_spawnattrs_setdisclaim()` (libsystem) is the documented
// escape hatch: when set on a posix_spawn attr block, the spawned child
// disclaims its parent's responsibility chain and TCC attributes requests
// to the child's own code identity. Browsers, Electron apps, and a number
// of native macOS tools use this exact pattern.
//
// This launcher does the bare minimum: posix_spawn the helper with the
// disclaim flag, forward common signals so Popen.terminate()/.kill() still
// reach the helper, and propagate the exit status back to Python.

#include <errno.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
extern int responsibility_spawnattrs_setdisclaim(posix_spawnattr_t *attrs,
                                                 int disclaim);

static volatile pid_t child_pid = 0;

// Grace period (seconds) between forwarding a termination signal and
// escalating to SIGKILL on the child. Picked to be well under the Python
// side's `_PROC_WAIT_TIMEOUT` so the escalation completes before Python
// itself escalates from SIGTERM to SIGKILL on the launcher.
#define CHILD_KILL_GRACE_SECONDS 2

static void escalate_to_sigkill(int sig) {
    (void)sig;
    if (child_pid > 0) {
        kill(child_pid, SIGKILL);
    }
}

static void forward_signal(int sig) {
    if (child_pid > 0) {
        kill(child_pid, sig);
        // If the child ignores the forwarded signal (wedged Swift helper,
        // for example), Python's later SIGKILL would only kill the
        // launcher and orphan the child. Arm an alarm so the launcher
        // SIGKILLs the child itself if it hasn't exited within the grace
        // period.
        signal(SIGALRM, escalate_to_sigkill);
        alarm(CHILD_KILL_GRACE_SECONDS);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "disclaim_launcher: missing target executable\n");
        return 2;
    }

    posix_spawnattr_t attrs;
    if (posix_spawnattr_init(&attrs) != 0) {
        perror("posix_spawnattr_init");
        return 1;
    }

    if (responsibility_spawnattrs_setdisclaim(&attrs, 1) != 0) {
        // Non-fatal: continue without disclaim. TCC attribution will fall
        // back to the parent process (the inherited-responsibility bug
        // this launcher exists to avoid), but capture functionality may
        // still work if the parent already holds the necessary grants.
        fprintf(stderr,
                "disclaim_launcher: setdisclaim failed (errno %d); "
                "TCC may attribute to parent\n",
                errno);
    }

    pid_t pid;
    int rc = posix_spawn(&pid, argv[1], NULL, &attrs, &argv[1], environ);
    posix_spawnattr_destroy(&attrs);
    if (rc != 0) {
        fprintf(stderr,
                "disclaim_launcher: posix_spawn(%s) failed: %s\n",
                argv[1], strerror(rc));
        return 1;
    }
    child_pid = pid;

    signal(SIGTERM, forward_signal);
    signal(SIGINT, forward_signal);
    signal(SIGHUP, forward_signal);
    signal(SIGQUIT, forward_signal);

    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            perror("waitpid");
            return 1;
        }
    }

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        signal(WTERMSIG(status), SIG_DFL);
        raise(WTERMSIG(status));
    }
    return 1;
}
