#include "ops/ir/parallel_compile.h"
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

/* Each "compile" writes its index into a per-index slot (no shared mutation),
 * so a correct run fills every slot exactly once regardless of ordering. */
typedef struct {
    atomic_int calls;
    int slot[256];
} Ctx;
static int mock_compile(int index, void* user) {
    Ctx* c = (Ctx*)user;
    atomic_fetch_add(&c->calls, 1);
    c->slot[index] = index + 1; /* non-zero marker */
    return 0;
}
static int fail_at_3(int index, void* user) {
    (void)user;
    return index == 3 ? 42 : 0;
}

static int check_all(Ctx* c, int n) {
    if (atomic_load(&c->calls) != n)
        return 0;
    for (int i = 0; i < n; i++)
        if (c->slot[i] != i + 1)
            return 0;
    return 1;
}

int main(void) {
    int n = 64, ok = 1;

    Ctx s;
    memset(&s, 0, sizeof s);
    ok &= (cml_compile_kernels(n, mock_compile, &s, false) == 0) && check_all(&s, n);
    printf("serial:   %s\n", ok ? "ok" : "FAIL");

    Ctx p;
    memset(&p, 0, sizeof p);
    int pr = (cml_compile_kernels(n, mock_compile, &p, true) == 0) && check_all(&p, n);
    ok &= pr;
    printf("parallel: %s\n", pr ? "ok" : "FAIL");

    /* A failing compile is surfaced (both modes). */
    ok &= (cml_compile_kernels(n, fail_at_3, NULL, false) == 42);
    ok &= (cml_compile_kernels(n, fail_at_3, NULL, true) == 42);
    printf("error-propagation: %s\n", ok ? "ok" : "FAIL");

    printf(ok ? "PASS\n" : "FAIL\n");
    return ok ? 0 : 1;
}
