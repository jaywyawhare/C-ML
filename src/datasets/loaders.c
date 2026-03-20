#define _POSIX_C_SOURCE 200809L
#include "datasets/loaders.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ctype.h>
#include <strings.h>

static int is_image_file(const char* name) {
    const char* ext = strrchr(name, '.');
    if (!ext) return 0;
    ext++;
    return (strcasecmp(ext, "ppm") == 0 ||
            strcasecmp(ext, "pgm") == 0 ||
            strcasecmp(ext, "raw") == 0 ||
            strcasecmp(ext, "bmp") == 0 ||
            strcasecmp(ext, "jpg") == 0 ||
            strcasecmp(ext, "jpeg") == 0 ||
            strcasecmp(ext, "png") == 0);
}

static int is_audio_file(const char* name) {
    const char* ext = strrchr(name, '.');
    if (!ext) return 0;
    ext++;
    return (strcasecmp(ext, "flac") == 0 ||
            strcasecmp(ext, "wav") == 0);
}

static int cmp_str(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

static int count_dir_entries(const char* path, int (*filter)(const char*)) {
    DIR* d = opendir(path);
    if (!d) return 0;
    int count = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (filter && !filter(ent->d_name)) continue;
        count++;
    }
    closedir(d);
    return count;
}

static int is_directory(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0 && S_ISDIR(st.st_mode));
}

static char** list_subdirs(const char* path, int* count) {
    DIR* d = opendir(path);
    if (!d) { *count = 0; return NULL; }

    int cap = 64;
    char** dirs = malloc(cap * sizeof(char*));
    *count = 0;

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        if (!is_directory(full)) continue;
        if (*count >= cap) {
            cap *= 2;
            dirs = realloc(dirs, cap * sizeof(char*));
        }
        dirs[(*count)++] = strdup(ent->d_name);
    }
    closedir(d);
    qsort(dirs, *count, sizeof(char*), cmp_str);
    return dirs;
}

static char** list_files_in_dir(const char* path, int (*filter)(const char*), int* count) {
    DIR* d = opendir(path);
    if (!d) { *count = 0; return NULL; }

    int cap = 256;
    char** files = malloc(cap * sizeof(char*));
    *count = 0;

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        if (filter && !filter(ent->d_name)) continue;
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", path, ent->d_name);
        struct stat st;
        if (stat(full, &st) != 0 || !S_ISREG(st.st_mode)) continue;
        if (*count >= cap) {
            cap *= 2;
            files = realloc(files, cap * sizeof(char*));
        }
        files[(*count)++] = strdup(full);
    }
    closedir(d);
    qsort(files, *count, sizeof(char*), cmp_str);
    return files;
}

static float* load_ppm_image(const char* path, int target_size, int* out_channels) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    char magic[4];
    if (!fgets(magic, sizeof(magic), f)) { fclose(f); return NULL; }

    int is_p6 = (magic[0] == 'P' && magic[1] == '6');
    int is_p5 = (magic[0] == 'P' && magic[1] == '5');
    if (!is_p6 && !is_p5) { fclose(f); return NULL; }

    int channels = is_p6 ? 3 : 1;
    *out_channels = channels;

    char line[256];
    int w = 0, h = 0, maxval = 0;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#') continue;
        if (w == 0) {
            if (sscanf(line, "%d %d", &w, &h) == 2) continue;
        }
        if (maxval == 0) {
            sscanf(line, "%d", &maxval);
            break;
        }
    }

    if (w <= 0 || h <= 0 || maxval <= 0) { fclose(f); return NULL; }

    size_t pixel_count = (size_t)w * h * channels;
    unsigned char* raw = malloc(pixel_count);
    if (!raw) { fclose(f); return NULL; }
    if (fread(raw, 1, pixel_count, f) != pixel_count) {
        free(raw);
        fclose(f);
        return NULL;
    }
    fclose(f);

    int ts = target_size > 0 ? target_size : h;
    float* out = malloc((size_t)ts * ts * channels * sizeof(float));
    if (!out) { free(raw); return NULL; }

    float scale_y = (float)h / ts;
    float scale_x = (float)w / ts;

    for (int y = 0; y < ts; y++) {
        for (int x = 0; x < ts; x++) {
            int sy = (int)(y * scale_y);
            int sx = (int)(x * scale_x);
            if (sy >= h) sy = h - 1;
            if (sx >= w) sx = w - 1;
            for (int c = 0; c < channels; c++) {
                float v = raw[(sy * w + sx) * channels + c] / (float)maxval;
                out[(y * ts + x) * channels + c] = v;
            }
        }
    }

    free(raw);
    return out;
}

CMLImageNetLoader* cml_imagenet_open(const char* dir_path, int image_size) {
    if (!dir_path) return NULL;

    int num_classes = 0;
    char** class_dirs = list_subdirs(dir_path, &num_classes);
    if (num_classes == 0) {
        LOG_ERROR("[loaders] No class subdirectories found in %s", dir_path);
        return NULL;
    }

    int total = 0;
    for (int c = 0; c < num_classes; c++) {
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", dir_path, class_dirs[c]);
        total += count_dir_entries(full, is_image_file);
    }

    if (total == 0) {
        LOG_ERROR("[loaders] No image files found in %s", dir_path);
        for (int i = 0; i < num_classes; i++) free(class_dirs[i]);
        free(class_dirs);
        return NULL;
    }

    CMLImageNetLoader* loader = calloc(1, sizeof(CMLImageNetLoader));
    loader->image_paths = malloc(total * sizeof(char*));
    loader->labels = malloc(total * sizeof(int));
    loader->num_classes = num_classes;
    loader->image_size = image_size > 0 ? image_size : 224;
    loader->num_samples = 0;

    for (int c = 0; c < num_classes; c++) {
        char class_path[2048];
        snprintf(class_path, sizeof(class_path), "%s/%s", dir_path, class_dirs[c]);

        int nfiles = 0;
        char** files = list_files_in_dir(class_path, is_image_file, &nfiles);
        for (int j = 0; j < nfiles; j++) {
            int idx = loader->num_samples++;
            loader->image_paths[idx] = files[j];
            loader->labels[idx] = c;
        }
        free(files);
    }

    LOG_INFO("[loaders] ImageNet: %d samples, %d classes from %s",
             loader->num_samples, num_classes, dir_path);

    for (int i = 0; i < num_classes; i++) free(class_dirs[i]);
    free(class_dirs);
    return loader;
}

Dataset* cml_imagenet_load_batch(CMLImageNetLoader* loader, int offset, int batch_size) {
    if (!loader || offset < 0 || offset >= loader->num_samples) return NULL;

    int actual_size = batch_size;
    if (offset + actual_size > loader->num_samples)
        actual_size = loader->num_samples - offset;

    int img_size = loader->image_size;
    int channels = 3;
    int feat = channels * img_size * img_size;

    float* X = calloc((size_t)actual_size * feat, sizeof(float));
    float* y = malloc((size_t)actual_size * sizeof(float));
    if (!X || !y) { free(X); free(y); return NULL; }

    int loaded = 0;
    for (int i = 0; i < actual_size; i++) {
        int idx = offset + i;
        int ch = 0;
        float* pixels = load_ppm_image(loader->image_paths[idx], img_size, &ch);
        if (pixels) {
            int src_feat = ch * img_size * img_size;
            if (src_feat <= feat)
                memcpy(X + (size_t)loaded * feat, pixels, src_feat * sizeof(float));
            free(pixels);
        }
        y[loaded] = (float)loader->labels[idx];
        loaded++;
    }

    Dataset* ds = dataset_from_arrays(X, y, loaded, feat, 1);
    if (ds) {
        ds->name = "imagenet_batch";
        ds->num_classes = loader->num_classes;
    }
    free(X);
    free(y);
    return ds;
}

void cml_imagenet_free(CMLImageNetLoader* loader) {
    if (!loader) return;
    for (int i = 0; i < loader->num_samples; i++)
        free(loader->image_paths[i]);
    free(loader->image_paths);
    free(loader->labels);
    free(loader);
}

static void collect_audio_recursive(const char* dir, char*** paths, char*** transcripts,
                                    int* count, int* cap) {
    DIR* d = opendir(dir);
    if (!d) return;

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", dir, ent->d_name);

        struct stat st;
        if (stat(full, &st) != 0) continue;

        if (S_ISDIR(st.st_mode)) {
            collect_audio_recursive(full, paths, transcripts, count, cap);
            continue;
        }

        if (!is_audio_file(ent->d_name)) continue;

        if (*count >= *cap) {
            *cap *= 2;
            *paths = realloc(*paths, *cap * sizeof(char*));
            *transcripts = realloc(*transcripts, *cap * sizeof(char*));
        }

        (*paths)[*count] = strdup(full);

        char txt_path[2048];
        snprintf(txt_path, sizeof(txt_path), "%s", full);
        char* ext = strrchr(txt_path, '.');
        if (ext) strcpy(ext, ".txt");

        char* transcript = NULL;
        FILE* tf = fopen(txt_path, "r");
        if (!tf) {
            char* last_slash = strrchr(txt_path, '/');
            if (last_slash) {
                char parent_dir[2048];
                size_t plen = last_slash - txt_path;
                memcpy(parent_dir, txt_path, plen);
                parent_dir[plen] = '\0';

                char* dash = strrchr(ent->d_name, '-');
                if (dash) {
                    char trans_file[4096];
                    char base[256];
                    strncpy(base, ent->d_name, sizeof(base) - 1);
                    base[sizeof(base) - 1] = '\0';
                    char* bd = strrchr(base, '-');
                    if (bd) *bd = '\0';
                    snprintf(trans_file, sizeof(trans_file), "%s/%s.trans.txt", parent_dir, base);
                    tf = fopen(trans_file, "r");
                }
            }
        }

        if (tf) {
            char line[4096];
            char* file_id = strrchr(ent->d_name, '/');
            file_id = file_id ? file_id + 1 : ent->d_name;
            char id_noext[256];
            strncpy(id_noext, file_id, sizeof(id_noext) - 1);
            id_noext[sizeof(id_noext) - 1] = '\0';
            char* id_dot = strrchr(id_noext, '.');
            if (id_dot) *id_dot = '\0';

            while (fgets(line, sizeof(line), tf)) {
                if (strncmp(line, id_noext, strlen(id_noext)) == 0) {
                    char* text = line + strlen(id_noext);
                    while (*text == ' ') text++;
                    size_t tlen = strlen(text);
                    while (tlen > 0 && (text[tlen - 1] == '\n' || text[tlen - 1] == '\r'))
                        tlen--;
                    transcript = strndup(text, tlen);
                    break;
                }
            }
            fclose(tf);
        }

        (*transcripts)[*count] = transcript ? transcript : strdup("");
        (*count)++;
    }
    closedir(d);
}

CMLLibriSpeechLoader* cml_librispeech_open(const char* dir_path) {
    if (!dir_path) return NULL;

    int cap = 1024;
    int count = 0;
    char** paths = malloc(cap * sizeof(char*));
    char** transcripts = malloc(cap * sizeof(char*));

    collect_audio_recursive(dir_path, &paths, &transcripts, &count, &cap);

    if (count == 0) {
        LOG_ERROR("[loaders] No audio files found in %s", dir_path);
        free(paths);
        free(transcripts);
        return NULL;
    }

    CMLLibriSpeechLoader* loader = calloc(1, sizeof(CMLLibriSpeechLoader));
    loader->audio_paths = paths;
    loader->transcripts = transcripts;
    loader->num_samples = count;
    loader->sample_rate = 16000;

    LOG_INFO("[loaders] LibriSpeech: %d samples from %s", count, dir_path);
    return loader;
}

void cml_librispeech_free(CMLLibriSpeechLoader* loader) {
    if (!loader) return;
    for (int i = 0; i < loader->num_samples; i++) {
        free(loader->audio_paths[i]);
        free(loader->transcripts[i]);
    }
    free(loader->audio_paths);
    free(loader->transcripts);
    free(loader);
}

static char* json_skip_ws(char* p) {
    while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
    return p;
}

static char* json_parse_string(char* p, char** out) {
    p = json_skip_ws(p);
    if (*p != '"') return NULL;
    p++;
    char* start = p;
    while (*p && *p != '"') {
        if (*p == '\\') p++;
        p++;
    }
    if (*p != '"') return NULL;
    *out = strndup(start, p - start);
    return p + 1;
}

static char* json_skip_value(char* p) {
    p = json_skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') { if (*p == '\\') p++; p++; }
        return *p == '"' ? p + 1 : p;
    }
    if (*p == '{') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } }
            p++;
        }
        return p;
    }
    if (*p == '[') {
        int depth = 1; p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } }
            p++;
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

static char* json_expect(char* p, char c) {
    p = json_skip_ws(p);
    return (*p == c) ? p + 1 : NULL;
}

CMLSQuADLoader* cml_squad_open(const char* json_path) {
    if (!json_path) return NULL;

    FILE* f = fopen(json_path, "r");
    if (!f) {
        LOG_ERROR("[loaders] Cannot open SQuAD file: %s", json_path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);

    char* buf = malloc(fsize + 1);
    if (!buf) { fclose(f); return NULL; }
    if ((long)fread(buf, 1, fsize, f) != fsize) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[fsize] = '\0';
    fclose(f);

    int cap = 4096;
    int count = 0;
    char** contexts = malloc(cap * sizeof(char*));
    char** questions = malloc(cap * sizeof(char*));
    char** answers = malloc(cap * sizeof(char*));
    int* answer_starts = malloc(cap * sizeof(int));

    /* Find "data" array */
    char* p = strstr(buf, "\"data\"");
    if (!p) goto done;
    p += 6;
    p = json_skip_ws(p);
    if (*p == ':') p++;
    p = json_expect(p, '[');
    if (!p) goto done;

    /* Iterate data array entries */
    while (p && *p) {
        p = json_skip_ws(p);
        if (*p == ']') break;
        if (*p == ',') p++;
        p = json_skip_ws(p);
        if (*p != '{') break;
        p++;

        /* Find "paragraphs" in this entry */
        char* para_start = strstr(p, "\"paragraphs\"");
        if (!para_start) { p = json_skip_value(p - 1); continue; }
        p = para_start + 12;
        p = json_skip_ws(p);
        if (*p == ':') p++;
        p = json_expect(p, '[');
        if (!p) break;

        /* Iterate paragraphs */
        while (p && *p) {
            p = json_skip_ws(p);
            if (*p == ']') { p++; break; }
            if (*p == ',') p++;
            p = json_skip_ws(p);
            if (*p != '{') break;
            p++;

            char* context = NULL;
            /* Find "context" */
            char* ctx_pos = strstr(p, "\"context\"");
            if (ctx_pos) {
                char* cp = ctx_pos + 9;
                cp = json_skip_ws(cp);
                if (*cp == ':') cp++;
                cp = json_parse_string(cp, &context);
            }

            /* Find "qas" array */
            char* qas_pos = strstr(p, "\"qas\"");
            if (qas_pos && context) {
                char* qp = qas_pos + 5;
                qp = json_skip_ws(qp);
                if (*qp == ':') qp++;
                qp = json_expect(qp, '[');
                if (!qp) { free(context); p = json_skip_value(p - 1); continue; }

                while (qp && *qp) {
                    qp = json_skip_ws(qp);
                    if (*qp == ']') { qp++; break; }
                    if (*qp == ',') qp++;
                    qp = json_skip_ws(qp);
                    if (*qp != '{') break;
                    qp++;

                    char* question = NULL;
                    char* answer = NULL;
                    int ans_start = 0;

                    char* q_pos = strstr(qp, "\"question\"");
                    if (q_pos) {
                        char* qq = q_pos + 10;
                        qq = json_skip_ws(qq);
                        if (*qq == ':') qq++;
                        json_parse_string(qq, &question);
                    }

                    char* ans_pos = strstr(qp, "\"answers\"");
                    if (ans_pos) {
                        char* ap = ans_pos + 9;
                        ap = json_skip_ws(ap);
                        if (*ap == ':') ap++;
                        ap = json_expect(ap, '[');
                        if (ap) {
                            ap = json_skip_ws(ap);
                            if (*ap == '{') {
                                ap++;
                                char* text_pos = strstr(ap, "\"text\"");
                                if (text_pos) {
                                    char* tp = text_pos + 6;
                                    tp = json_skip_ws(tp);
                                    if (*tp == ':') tp++;
                                    json_parse_string(tp, &answer);
                                }
                                char* start_pos = strstr(ap, "\"answer_start\"");
                                if (start_pos) {
                                    char* sp = start_pos + 14;
                                    sp = json_skip_ws(sp);
                                    if (*sp == ':') sp++;
                                    sp = json_skip_ws(sp);
                                    ans_start = (int)strtol(sp, NULL, 10);
                                }
                            }
                        }
                    }

                    if (question) {
                        if (count >= cap) {
                            cap *= 2;
                            contexts = realloc(contexts, cap * sizeof(char*));
                            questions = realloc(questions, cap * sizeof(char*));
                            answers = realloc(answers, cap * sizeof(char*));
                            answer_starts = realloc(answer_starts, cap * sizeof(int));
                        }
                        contexts[count] = strdup(context);
                        questions[count] = question;
                        answers[count] = answer ? answer : strdup("");
                        answer_starts[count] = ans_start;
                        count++;
                    } else {
                        free(question);
                        free(answer);
                    }

                    /* Skip to end of this qa object */
                    int depth = 1;
                    while (*qp && depth > 0) {
                        if (*qp == '{') depth++;
                        else if (*qp == '}') depth--;
                        else if (*qp == '"') { qp++; while (*qp && *qp != '"') { if (*qp == '\\') qp++; qp++; } }
                        qp++;
                    }
                }
            } else {
                free(context);
            }

            /* Skip to end of this paragraph object */
            int depth = 1;
            while (*p && depth > 0) {
                if (*p == '{') depth++;
                else if (*p == '}') depth--;
                else if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } }
                p++;
            }
        }

        /* Skip to end of data entry */
        p = json_skip_ws(p);
        while (*p && *p != '}') {
            if (*p == '"') { p++; while (*p && *p != '"') { if (*p == '\\') p++; p++; } p++; }
            else p++;
        }
        if (*p == '}') p++;
    }

done:
    free(buf);

    if (count == 0) {
        LOG_ERROR("[loaders] No QA pairs found in %s", json_path);
        free(contexts); free(questions); free(answers); free(answer_starts);
        return NULL;
    }

    CMLSQuADLoader* loader = calloc(1, sizeof(CMLSQuADLoader));
    loader->contexts = contexts;
    loader->questions = questions;
    loader->answers = answers;
    loader->answer_starts = answer_starts;
    loader->num_samples = count;

    LOG_INFO("[loaders] SQuAD: %d QA pairs from %s", count, json_path);
    return loader;
}

void cml_squad_free(CMLSQuADLoader* loader) {
    if (!loader) return;
    for (int i = 0; i < loader->num_samples; i++) {
        free(loader->contexts[i]);
        free(loader->questions[i]);
        free(loader->answers[i]);
    }
    free(loader->contexts);
    free(loader->questions);
    free(loader->answers);
    free(loader->answer_starts);
    free(loader);
}

/* NIfTI-1 header (simplified, 348 bytes) */
typedef struct {
    int sizeof_hdr;
    char pad1[36];
    short dim[8];
    char pad2[14];
    short datatype;
    short bitpix;
    char pad3[38];
    float pixdim[8];
    float vox_offset;
    char pad4[120];
    char magic[4];
} NIfTI1Header;

static float* nifti_read_volume(const char* path, int* dims_out, int* ndim_out) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    NIfTI1Header hdr;
    if (fread(&hdr, 1, sizeof(hdr), f) != sizeof(hdr)) {
        fclose(f);
        return NULL;
    }

    if (hdr.sizeof_hdr != 348) {
        fclose(f);
        return NULL;
    }

    int ndim = hdr.dim[0];
    if (ndim < 1 || ndim > 7) { fclose(f); return NULL; }

    size_t nvox = 1;
    for (int i = 1; i <= ndim; i++) {
        dims_out[i - 1] = hdr.dim[i];
        nvox *= (size_t)hdr.dim[i];
    }
    *ndim_out = ndim;

    long offset = (long)hdr.vox_offset;
    if (offset < 348) offset = 348;
    fseek(f, offset, SEEK_SET);

    float* data = malloc(nvox * sizeof(float));
    if (!data) { fclose(f); return NULL; }

    if (hdr.datatype == 16) { /* FLOAT32 */
        if (fread(data, sizeof(float), nvox, f) != nvox) {
            free(data); fclose(f); return NULL;
        }
    } else if (hdr.datatype == 4) { /* INT16 */
        short* raw = malloc(nvox * sizeof(short));
        if (!raw || fread(raw, sizeof(short), nvox, f) != nvox) {
            free(raw); free(data); fclose(f); return NULL;
        }
        for (size_t i = 0; i < nvox; i++) data[i] = (float)raw[i];
        free(raw);
    } else if (hdr.datatype == 2) { /* UINT8 */
        unsigned char* raw = malloc(nvox);
        if (!raw || fread(raw, 1, nvox, f) != nvox) {
            free(raw); free(data); fclose(f); return NULL;
        }
        for (size_t i = 0; i < nvox; i++) data[i] = (float)raw[i];
        free(raw);
    } else if (hdr.datatype == 8) { /* INT32 */
        int* raw = malloc(nvox * sizeof(int));
        if (!raw || fread(raw, sizeof(int), nvox, f) != nvox) {
            free(raw); free(data); fclose(f); return NULL;
        }
        for (size_t i = 0; i < nvox; i++) data[i] = (float)raw[i];
        free(raw);
    } else {
        free(data); fclose(f); return NULL;
    }

    fclose(f);
    return data;
}

static int is_kits19_case(const char* name) {
    return (strncmp(name, "case_", 5) == 0 && strlen(name) == 10);
}

CMLKiTS19Loader* cml_kits19_open(const char* data_dir) {
    if (!data_dir) return NULL;

    int cap = 256, count = 0;
    char** dirs = malloc(cap * sizeof(char*));

    DIR* d = opendir(data_dir);
    if (!d) { free(dirs); return NULL; }

    struct dirent* ent;
    while ((ent = readdir(d)) != NULL) {
        if (!is_kits19_case(ent->d_name)) continue;
        char full[2048];
        snprintf(full, sizeof(full), "%s/%s", data_dir, ent->d_name);
        if (!is_directory(full)) continue;
        if (count >= cap) {
            cap *= 2;
            dirs = realloc(dirs, cap * sizeof(char*));
        }
        dirs[count++] = strdup(full);
    }
    closedir(d);

    if (count == 0) {
        LOG_ERROR("[loaders] No KiTS19 case directories found in %s", data_dir);
        free(dirs);
        return NULL;
    }

    qsort(dirs, count, sizeof(char*), cmp_str);

    CMLKiTS19Loader* loader = calloc(1, sizeof(CMLKiTS19Loader));
    loader->case_dirs = dirs;
    loader->num_cases = count;

    LOG_INFO("[loaders] KiTS19: %d cases from %s", count, data_dir);
    return loader;
}

void cml_kits19_free(CMLKiTS19Loader* loader) {
    if (!loader) return;
    for (int i = 0; i < loader->num_cases; i++)
        free(loader->case_dirs[i]);
    free(loader->case_dirs);
    free(loader);
}

int cml_kits19_load_case(CMLKiTS19Loader* loader, int case_idx,
                         Tensor** volume, Tensor** segmentation) {
    if (!loader || case_idx < 0 || case_idx >= loader->num_cases) return -1;
    if (!volume || !segmentation) return -1;

    char vol_path[2048], seg_path[2048];
    snprintf(vol_path, sizeof(vol_path), "%s/imaging.nii", loader->case_dirs[case_idx]);
    snprintf(seg_path, sizeof(seg_path), "%s/segmentation.nii", loader->case_dirs[case_idx]);

    int vol_dims[7], vol_ndim = 0;
    float* vol_data = nifti_read_volume(vol_path, vol_dims, &vol_ndim);
    if (!vol_data) {
        LOG_ERROR("[loaders] Failed to load volume: %s", vol_path);
        return -1;
    }

    int seg_dims[7], seg_ndim = 0;
    float* seg_data = nifti_read_volume(seg_path, seg_dims, &seg_ndim);
    if (!seg_data) {
        LOG_ERROR("[loaders] Failed to load segmentation: %s", seg_path);
        free(vol_data);
        return -1;
    }

    *volume = tensor_from_data(vol_data, vol_dims, vol_ndim, NULL);
    *segmentation = tensor_from_data(seg_data, seg_dims, seg_ndim, NULL);

    free(vol_data);
    free(seg_data);

    if (!*volume || !*segmentation) return -1;
    return 0;
}

static int is_text_file(const char* name) {
    const char* ext = strrchr(name, '.');
    if (!ext) return 0;
    ext++;
    return (strcasecmp(ext, "txt") == 0 ||
            strcasecmp(ext, "text") == 0 ||
            strcasecmp(ext, "article") == 0);
}

CMLOpenImagesLoader* cml_openimages_open(const char* images_dir, const char* annotations_csv) {
    if (!images_dir || !annotations_csv) return NULL;

    int count = 0;
    char** files = list_files_in_dir(images_dir, is_image_file, &count);
    if (count == 0) {
        LOG_ERROR("[loaders] No images found in %s", images_dir);
        free(files);
        return NULL;
    }

    char** ids = malloc(count * sizeof(char*));
    for (int i = 0; i < count; i++) {
        const char* base = strrchr(files[i], '/');
        base = base ? base + 1 : files[i];
        char* dot = strrchr(base, '.');
        ids[i] = dot ? strndup(base, dot - base) : strdup(base);
        free(files[i]);
    }
    free(files);

    CMLOpenImagesLoader* loader = calloc(1, sizeof(CMLOpenImagesLoader));
    loader->image_ids = ids;
    loader->num_images = count;
    loader->images_dir = strdup(images_dir);
    loader->annotations_path = strdup(annotations_csv);

    LOG_INFO("[loaders] OpenImages: %d images from %s", count, images_dir);
    return loader;
}

void cml_openimages_free(CMLOpenImagesLoader* loader) {
    if (!loader) return;
    for (int i = 0; i < loader->num_images; i++)
        free(loader->image_ids[i]);
    free(loader->image_ids);
    free(loader->images_dir);
    free(loader->annotations_path);
    free(loader);
}

CMLWikipediaLoader* cml_wikipedia_open(const char* dump_dir) {
    if (!dump_dir) return NULL;

    int count = 0;
    char** files = list_files_in_dir(dump_dir, is_text_file, &count);
    if (count == 0) {
        LOG_ERROR("[loaders] No text files found in %s", dump_dir);
        free(files);
        return NULL;
    }

    size_t total = 0;
    for (int i = 0; i < count; i++) {
        struct stat st;
        if (stat(files[i], &st) == 0)
            total += st.st_size;
    }

    CMLWikipediaLoader* loader = calloc(1, sizeof(CMLWikipediaLoader));
    loader->article_paths = files;
    loader->num_articles = count;
    loader->total_bytes = total;

    LOG_INFO("[loaders] Wikipedia: %d articles, %.1f MB from %s",
             count, total / (1024.0 * 1024.0), dump_dir);
    return loader;
}

void cml_wikipedia_free(CMLWikipediaLoader* loader) {
    if (!loader) return;
    for (int i = 0; i < loader->num_articles; i++)
        free(loader->article_paths[i]);
    free(loader->article_paths);
    free(loader);
}

int cml_wikipedia_read_chunk(CMLWikipediaLoader* loader, int article_idx,
                             char* buf, size_t buf_size, size_t* bytes_read) {
    if (!loader || article_idx < 0 || article_idx >= loader->num_articles)
        return -1;
    if (!buf || buf_size == 0 || !bytes_read) return -1;

    FILE* f = fopen(loader->article_paths[article_idx], "r");
    if (!f) return -1;

    *bytes_read = fread(buf, 1, buf_size - 1, f);
    buf[*bytes_read] = '\0';
    fclose(f);
    return 0;
}

Dataset* cml_load_image_folder(const char* dir_path, int image_size) {
    CMLImageNetLoader* loader = cml_imagenet_open(dir_path, image_size);
    if (!loader) return NULL;

    Dataset* ds = cml_imagenet_load_batch(loader, 0, loader->num_samples);
    cml_imagenet_free(loader);
    return ds;
}
