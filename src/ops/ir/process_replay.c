#include "ops/ir/process_replay.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#include <stdbool.h>
#include <stdint.h>

#define FNV_OFFSET_BASIS 0xcbf29ce484222325ULL
#define FNV_PRIME 0x100000001b3ULL
#define HEADER_MAGIC "CMLPR001"
#define KERNEL_FILE_EXT ".kernel"

static pthread_mutex_t g_replay_lock = PTHREAD_MUTEX_INITIALIZER;
static bool g_replay_enabled = false;
static char g_output_dir[4096];

static uint64_t fnv1a(const void* data, size_t len) {
    uint64_t hash = FNV_OFFSET_BASIS;
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

static void ensure_dir(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) {
        mkdir(path, 0755);
    }
}

void cml_process_replay_enable(const char* output_dir) {
    pthread_mutex_lock(&g_replay_lock);

    if (!output_dir || output_dir[0] == '\0') {
        pthread_mutex_unlock(&g_replay_lock);
        return;
    }

    snprintf(g_output_dir, sizeof(g_output_dir), "%s", output_dir);
    ensure_dir(g_output_dir);
    g_replay_enabled = true;

    LOG_INFO("Process replay recording to %s", g_output_dir);
    pthread_mutex_unlock(&g_replay_lock);
}

void cml_process_replay_disable(void) {
    pthread_mutex_lock(&g_replay_lock);
    g_replay_enabled = false;
    pthread_mutex_unlock(&g_replay_lock);
}

static void check_env_init(void) {
    static bool checked = false;
    if (checked) return;
    checked = true;

    const char* env = getenv("CML_PROCESS_REPLAY");
    if (!env || env[0] == '\0') return;

    const char* dir = getenv("CML_PROCESS_REPLAY_DIR");
    if (!dir || dir[0] == '\0') dir = "/tmp/cml_process_replay";

    cml_process_replay_enable(dir);
}

void cml_process_replay_record(const char* kernel_name, const char* source, size_t source_len) {
    check_env_init();

    pthread_mutex_lock(&g_replay_lock);
    if (!g_replay_enabled) {
        pthread_mutex_unlock(&g_replay_lock);
        return;
    }

    uint64_t hash = fnv1a(source, source_len);

    char path[4224];
    snprintf(path, sizeof(path), "%s/%016llx%s",
             g_output_dir, (unsigned long long)hash, KERNEL_FILE_EXT);

    FILE* f = fopen(path, "wb");
    if (!f) {
        LOG_WARNING("Process replay: failed to write %s", path);
        pthread_mutex_unlock(&g_replay_lock);
        return;
    }

    time_t now = time(NULL);
    fprintf(f, "%s\n", HEADER_MAGIC);
    fprintf(f, "name: %s\n", kernel_name ? kernel_name : "unknown");
    fprintf(f, "timestamp: %ld\n", (long)now);
    fprintf(f, "source_len: %zu\n", source_len);
    fprintf(f, "---\n");
    fwrite(source, 1, source_len, f);
    fclose(f);

    LOG_DEBUG("Process replay: recorded %s -> %016llx", kernel_name, (unsigned long long)hash);
    pthread_mutex_unlock(&g_replay_lock);
}

static int read_kernel_file(const char* path, char** out_source, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long total = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (total <= 0) {
        fclose(f);
        return -1;
    }

    char* buf = malloc((size_t)total + 1);
    if (!buf) {
        fclose(f);
        return -1;
    }

    size_t read = fread(buf, 1, (size_t)total, f);
    fclose(f);
    buf[read] = '\0';

    char* sep = strstr(buf, "---\n");
    if (!sep) {
        free(buf);
        return -1;
    }

    char* body = sep + 4;
    size_t body_len = read - (size_t)(body - buf);

    *out_source = malloc(body_len + 1);
    if (!*out_source) {
        free(buf);
        return -1;
    }
    memcpy(*out_source, body, body_len);
    (*out_source)[body_len] = '\0';
    *out_len = body_len;

    free(buf);
    return 0;
}

int cml_process_replay_compare(const char* output_dir, const char* baseline_dir) {
    if (!output_dir || !baseline_dir) return -1;

    DIR* dir = opendir(baseline_dir);
    if (!dir) {
        LOG_ERROR("Process replay: cannot open baseline dir %s", baseline_dir);
        return -1;
    }

    int mismatches = 0;
    int compared = 0;
    struct dirent* ent;

    while ((ent = readdir(dir)) != NULL) {
        size_t namelen = strlen(ent->d_name);
        size_t extlen = strlen(KERNEL_FILE_EXT);
        if (namelen <= extlen) continue;
        if (strcmp(ent->d_name + namelen - extlen, KERNEL_FILE_EXT) != 0) continue;

        char baseline_path[4224];
        char output_path[4224];
        snprintf(baseline_path, sizeof(baseline_path), "%s/%s", baseline_dir, ent->d_name);
        snprintf(output_path, sizeof(output_path), "%s/%s", output_dir, ent->d_name);

        char* baseline_src = NULL;
        char* output_src = NULL;
        size_t baseline_len = 0;
        size_t output_len = 0;

        if (read_kernel_file(baseline_path, &baseline_src, &baseline_len) != 0) {
            LOG_WARNING("Process replay: cannot read baseline %s", baseline_path);
            continue;
        }

        if (read_kernel_file(output_path, &output_src, &output_len) != 0) {
            LOG_ERROR("Process replay: missing output for %s", ent->d_name);
            free(baseline_src);
            mismatches++;
            continue;
        }

        compared++;

        if (baseline_len != output_len || memcmp(baseline_src, output_src, baseline_len) != 0) {
            LOG_ERROR("Process replay: MISMATCH in %s", ent->d_name);
            mismatches++;
        }

        free(baseline_src);
        free(output_src);
    }

    closedir(dir);

    if (mismatches == 0) {
        LOG_INFO("Process replay: %d kernels compared, all match", compared);
    } else {
        LOG_ERROR("Process replay: %d/%d kernels mismatched", mismatches, compared);
    }

    return mismatches;
}
