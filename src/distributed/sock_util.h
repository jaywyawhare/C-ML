/* POSIX socket helpers shared by the TCP-based comm backends (Gloo, NCCL
 * bootstrap, IB). Not built on Windows, where those backends are stubbed. */

#ifndef CML_DISTRIBUTED_SOCK_UTIL_H
#define CML_DISTRIBUTED_SOCK_UTIL_H

#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>

/**
 * @brief Connect a TCP stream socket to @p addr, retrying while the peer is
 *        not yet listening.
 *
 * Every attempt uses a fresh socket: after a failed connect() the socket is in
 * an unspecified state, and on macOS/BSD every later connect() on it fails too
 * (Linux happens to tolerate the reuse), which turned a startup race into a
 * hard timeout.
 *
 * @param addr     Peer address.
 * @param len      Size of @p addr.
 * @param retries  Maximum number of attempts.
 * @param delay_us Sleep between attempts, in microseconds.
 * @return A connected socket fd, or -1 if every attempt failed (errno is
 *         preserved from the last failure).
 */
static inline int cml_sock_connect_retry(const struct sockaddr* addr, socklen_t len, int retries,
                                         unsigned delay_us) {
    for (int i = 0; i < retries; i++) {
        int fd = socket(addr->sa_family, SOCK_STREAM, 0);
        if (fd < 0)
            return -1;
        if (connect(fd, addr, len) == 0)
            return fd;
        int saved = errno;
        close(fd);
        errno = saved;
        usleep(delay_us);
    }
    return -1;
}

#endif /* CML_DISTRIBUTED_SOCK_UTIL_H */
