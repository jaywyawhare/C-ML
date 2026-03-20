#define _POSIX_C_SOURCE 200809L
#include "ops/ir/disk_cache.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <time.h>
#include <errno.h>

typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;

#define SQLITE_OK         0
#define SQLITE_ROW        100
#define SQLITE_DONE       101
#define SQLITE_TRANSIENT  ((void(*)(void*))-1)

typedef int (*fn_sqlite3_open)(const char*, sqlite3**);
typedef int (*fn_sqlite3_close)(sqlite3*);
typedef int (*fn_sqlite3_exec)(sqlite3*, const char*, void*, void*, char**);
typedef int (*fn_sqlite3_prepare_v2)(sqlite3*, const char*, int, sqlite3_stmt**, const char**);
typedef int (*fn_sqlite3_bind_int64)(sqlite3_stmt*, int, int64_t);
typedef int (*fn_sqlite3_bind_blob)(sqlite3_stmt*, int, const void*, int, void(*)(void*));
typedef int (*fn_sqlite3_step)(sqlite3_stmt*);
typedef const void* (*fn_sqlite3_column_blob)(sqlite3_stmt*, int);
typedef int (*fn_sqlite3_column_int)(sqlite3_stmt*, int);
typedef int (*fn_sqlite3_column_bytes)(sqlite3_stmt*, int);
typedef int (*fn_sqlite3_finalize)(sqlite3_stmt*);
typedef void (*fn_sqlite3_free)(void*);

static struct {
    void* handle;
    fn_sqlite3_open       open;
    fn_sqlite3_close      close;
    fn_sqlite3_exec       exec;
    fn_sqlite3_prepare_v2 prepare_v2;
    fn_sqlite3_bind_int64 bind_int64;
    fn_sqlite3_bind_blob  bind_blob;
    fn_sqlite3_step       step;
    fn_sqlite3_column_blob column_blob;
    fn_sqlite3_column_int  column_int;
    fn_sqlite3_column_bytes column_bytes;
    fn_sqlite3_finalize   finalize;
    fn_sqlite3_free       free_fn;
    bool loaded;
} sql = {0};

struct CMLDiskCache {
    sqlite3* db;
    sqlite3_stmt* stmt_put;
    sqlite3_stmt* stmt_get;
    sqlite3_stmt* stmt_has;
    sqlite3_stmt* stmt_count;
};

static int load_sqlite(void) {
    if (sql.loaded) return 0;

    const char* names[] = {
        "libsqlite3.so.0", "libsqlite3.so", "libsqlite3.dylib", NULL
    };

    for (int i = 0; names[i]; i++) {
        sql.handle = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
        if (sql.handle) break;
    }

    if (!sql.handle) {
        LOG_WARNING("SQLite3 not found: disk cache unavailable");
        return -1;
    }

#define LOAD_SYM(name) do { \
    *(void**)&sql.name = dlsym(sql.handle, "sqlite3_" #name); \
    if (!sql.name) { \
        LOG_ERROR("Failed to load sqlite3_%s", #name); \
        dlclose(sql.handle); \
        sql.handle = NULL; \
        return -1; \
    } \
} while(0)

    LOAD_SYM(open);
    LOAD_SYM(close);
    LOAD_SYM(exec);
    LOAD_SYM(prepare_v2);
    LOAD_SYM(bind_int64);
    LOAD_SYM(bind_blob);
    LOAD_SYM(step);
    LOAD_SYM(column_blob);
    LOAD_SYM(column_int);
    LOAD_SYM(column_bytes);
    LOAD_SYM(finalize);

    *(void**)&sql.free_fn = dlsym(sql.handle, "sqlite3_free");

#undef LOAD_SYM

    sql.loaded = true;
    return 0;
}

static int mkdirs(const char* path) {
    char tmp[4096];
    size_t len = strlen(path);
    if (len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);

    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    return 0;
}

static char* default_path(void) {
    const char* cache_dir = getenv("CML_CACHE_DIR");
    if (cache_dir) {
        size_t len = strlen(cache_dir) + 32;
        char* buf = malloc(len);
        if (!buf) return NULL;
        snprintf(buf, len, "%s/kernels.db", cache_dir);
        return buf;
    }

    const char* home = getenv("HOME");
    if (!home) home = "/tmp";

    size_t len = strlen(home) + 32;
    char* buf = malloc(len);
    if (!buf) return NULL;
    snprintf(buf, len, "%s/.cache/cml/kernels.db", home);
    return buf;
}

bool cml_disk_cache_enabled(void) {
    const char* env = getenv("CML_DISK_CACHE");
    if (!env) return false;
    return env[0] == '1' || env[0] == 'y' || env[0] == 'Y';
}

CMLDiskCache* cml_disk_cache_open(const char* path) {
    if (load_sqlite() != 0) return NULL;

    char* alloc_path = NULL;
    if (!path) {
        alloc_path = default_path();
        if (!alloc_path) return NULL;
        path = alloc_path;
    }

    mkdirs(path);

    CMLDiskCache* cache = calloc(1, sizeof(CMLDiskCache));
    if (!cache) {
        free(alloc_path);
        return NULL;
    }

    if (sql.open(path, &cache->db) != SQLITE_OK) {
        LOG_ERROR("Failed to open disk cache: %s", path);
        free(cache);
        free(alloc_path);
        return NULL;
    }

    free(alloc_path);

    char* err = NULL;
    sql.exec(cache->db, "PRAGMA journal_mode=WAL", NULL, NULL, &err);
    if (err && sql.free_fn) sql.free_fn(err);

    err = NULL;
    sql.exec(cache->db,
        "CREATE TABLE IF NOT EXISTS kernels ("
        "hash INTEGER PRIMARY KEY,"
        "data BLOB,"
        "size INTEGER,"
        "created_at INTEGER"
        ")",
        NULL, NULL, &err);
    if (err) {
        LOG_ERROR("Failed to create table: %s", err);
        if (sql.free_fn) sql.free_fn(err);
        sql.close(cache->db);
        free(cache);
        return NULL;
    }

    const char* sql_put = "INSERT OR REPLACE INTO kernels (hash, data, size, created_at) VALUES (?, ?, ?, ?)";
    const char* sql_get = "SELECT data, size FROM kernels WHERE hash = ?";
    const char* sql_has = "SELECT 1 FROM kernels WHERE hash = ?";
    const char* sql_cnt = "SELECT COUNT(*) FROM kernels";

    if (sql.prepare_v2(cache->db, sql_put, -1, &cache->stmt_put, NULL) != SQLITE_OK ||
        sql.prepare_v2(cache->db, sql_get, -1, &cache->stmt_get, NULL) != SQLITE_OK ||
        sql.prepare_v2(cache->db, sql_has, -1, &cache->stmt_has, NULL) != SQLITE_OK ||
        sql.prepare_v2(cache->db, sql_cnt, -1, &cache->stmt_count, NULL) != SQLITE_OK) {
        LOG_ERROR("Failed to prepare statements");
        cml_disk_cache_close(cache);
        return NULL;
    }

    return cache;
}

void cml_disk_cache_close(CMLDiskCache* cache) {
    if (!cache) return;

    if (cache->stmt_put) sql.finalize(cache->stmt_put);
    if (cache->stmt_get) sql.finalize(cache->stmt_get);
    if (cache->stmt_has) sql.finalize(cache->stmt_has);
    if (cache->stmt_count) sql.finalize(cache->stmt_count);
    if (cache->db) sql.close(cache->db);

    free(cache);
}

typedef int (*fn_sqlite3_reset)(sqlite3_stmt*);
typedef int (*fn_sqlite3_clear_bindings)(sqlite3_stmt*);

static fn_sqlite3_reset get_reset(void) {
    static fn_sqlite3_reset fn = NULL;
    if (!fn && sql.handle)
        *(void**)&fn = dlsym(sql.handle, "sqlite3_reset");
    return fn;
}

static fn_sqlite3_clear_bindings get_clear_bindings(void) {
    static fn_sqlite3_clear_bindings fn = NULL;
    if (!fn && sql.handle)
        *(void**)&fn = dlsym(sql.handle, "sqlite3_clear_bindings");
    return fn;
}

static void reset_stmt(sqlite3_stmt* stmt) {
    fn_sqlite3_reset rst = get_reset();
    fn_sqlite3_clear_bindings clr = get_clear_bindings();
    if (rst) rst(stmt);
    if (clr) clr(stmt);
}

int cml_disk_cache_put(CMLDiskCache* cache, uint64_t hash, const void* data, size_t size) {
    if (!cache || !data || size == 0) return -1;

    reset_stmt(cache->stmt_put);

    sql.bind_int64(cache->stmt_put, 1, (int64_t)hash);
    sql.bind_blob(cache->stmt_put, 2, data, (int)size, SQLITE_TRANSIENT);
    sql.bind_int64(cache->stmt_put, 3, (int64_t)size);
    sql.bind_int64(cache->stmt_put, 4, (int64_t)time(NULL));

    int rc = sql.step(cache->stmt_put);
    reset_stmt(cache->stmt_put);

    return (rc == SQLITE_DONE) ? 0 : -1;
}

int cml_disk_cache_get(CMLDiskCache* cache, uint64_t hash, void** out_data, size_t* out_size) {
    if (!cache || !out_data || !out_size) return -1;

    *out_data = NULL;
    *out_size = 0;

    reset_stmt(cache->stmt_get);
    sql.bind_int64(cache->stmt_get, 1, (int64_t)hash);

    int rc = sql.step(cache->stmt_get);
    if (rc != SQLITE_ROW) {
        reset_stmt(cache->stmt_get);
        return -1;
    }

    const void* blob = sql.column_blob(cache->stmt_get, 0);
    int blob_size = sql.column_bytes(cache->stmt_get, 0);

    if (!blob || blob_size <= 0) {
        reset_stmt(cache->stmt_get);
        return -1;
    }

    void* buf = malloc((size_t)blob_size);
    if (!buf) {
        reset_stmt(cache->stmt_get);
        return -1;
    }

    memcpy(buf, blob, (size_t)blob_size);
    *out_data = buf;
    *out_size = (size_t)blob_size;

    reset_stmt(cache->stmt_get);
    return 0;
}

bool cml_disk_cache_has(CMLDiskCache* cache, uint64_t hash) {
    if (!cache) return false;

    reset_stmt(cache->stmt_has);
    sql.bind_int64(cache->stmt_has, 1, (int64_t)hash);

    int rc = sql.step(cache->stmt_has);
    reset_stmt(cache->stmt_has);

    return rc == SQLITE_ROW;
}

int cml_disk_cache_count(CMLDiskCache* cache) {
    if (!cache) return 0;

    reset_stmt(cache->stmt_count);
    int rc = sql.step(cache->stmt_count);

    int count = 0;
    if (rc == SQLITE_ROW)
        count = sql.column_int(cache->stmt_count, 0);

    reset_stmt(cache->stmt_count);
    return count;
}

int cml_disk_cache_clear(CMLDiskCache* cache) {
    if (!cache) return -1;

    char* err = NULL;
    sql.exec(cache->db, "DELETE FROM kernels", NULL, NULL, &err);
    if (err) {
        LOG_ERROR("Failed to clear disk cache: %s", err);
        if (sql.free_fn) sql.free_fn(err);
        return -1;
    }
    return 0;
}
