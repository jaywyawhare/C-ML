/**
 * @file csv_parser.c
 * @brief CSV parser with auto-detection of headers, delimiters, and string labels
 */

#define _POSIX_C_SOURCE 200809L
#include "datasets/datasets.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_LINE 8192
#define MAX_COLS 256
#define MAX_LABELS 64

static int is_numeric(const char* s) {
    if (!s || !*s) return 0;
    while (*s == ' ') s++;
    if (*s == '-' || *s == '+') s++;
    int has_digit = 0;
    while (*s) {
        if (*s == '.' || *s == 'e' || *s == 'E' || *s == '+' || *s == '-') {
            s++;
            continue;
        }
        if (isdigit((unsigned char)*s)) {
            has_digit = 1;
            s++;
        } else if (*s == '\r' || *s == '\n' || *s == ' ') {
            s++;
        } else {
            return 0;
        }
    }
    return has_digit;
}

static char detect_delimiter(const char* line) {
    int commas = 0, semicolons = 0, tabs = 0;
    for (const char* p = line; *p; p++) {
        if (*p == ',') commas++;
        else if (*p == ';') semicolons++;
        else if (*p == '\t') tabs++;
    }
    if (tabs > commas && tabs > semicolons) return '\t';
    if (semicolons > commas) return ';';
    return ',';
}

/* Split a line by delimiter, return field count. Fields written into fields[] */
static int split_line(char* line, char delim, char** fields, int max_fields) {
    int count = 0;
    char* p = line;

    /* Strip trailing newline/carriage return */
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        line[--len] = '\0';

    if (len == 0) return 0;

    while (p && count < max_fields) {
        fields[count++] = p;
        char* next = strchr(p, delim);
        if (next) {
            *next = '\0';
            p = next + 1;
        } else {
            break;
        }
    }
    return count;
}

typedef struct {
    char labels[MAX_LABELS][128];
    int count;
} LabelMap;

static int label_map_get_or_add(LabelMap* lm, const char* s) {
    /* Trim whitespace */
    while (*s == ' ') s++;
    char trimmed[128];
    strncpy(trimmed, s, 127);
    trimmed[127] = '\0';
    size_t tlen = strlen(trimmed);
    while (tlen > 0 && (trimmed[tlen - 1] == ' ' || trimmed[tlen - 1] == '\r' || trimmed[tlen - 1] == '\n'))
        trimmed[--tlen] = '\0';

    if (tlen == 0) return -1;

    for (int i = 0; i < lm->count; i++) {
        if (strcmp(lm->labels[i], trimmed) == 0)
            return i;
    }
    if (lm->count >= MAX_LABELS) return -1;
    strcpy(lm->labels[lm->count], trimmed);
    return lm->count++;
}

int cml_csv_parse(const char* filepath, int target_col,
                  float** X_out, float** y_out,
                  int* num_samples, int* num_features, int* num_classes,
                  char*** class_names_out) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        LOG_ERROR("[csv] Cannot open: %s", filepath);
        return -1;
    }

    char line[MAX_LINE];
    char* fields[MAX_COLS];

    /* Read first line to detect delimiter and header */
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    char delim = detect_delimiter(line);

    /* Detect if first line is a header (first field non-numeric) */
    char first_line_copy[MAX_LINE];
    strncpy(first_line_copy, line, MAX_LINE - 1);
    first_line_copy[MAX_LINE - 1] = '\0';

    char header_check[MAX_LINE];
    strncpy(header_check, line, MAX_LINE - 1);
    header_check[MAX_LINE - 1] = '\0';
    int hcount = split_line(header_check, delim, fields, MAX_COLS);
    int has_header = (hcount > 0 && !is_numeric(fields[0]));

    /* Count total columns from first data line */
    int total_cols;
    char data_line[MAX_LINE];
    if (has_header) {
        if (!fgets(data_line, sizeof(data_line), f)) {
            fclose(f);
            return -1;
        }
        char count_buf[MAX_LINE];
        strncpy(count_buf, data_line, MAX_LINE - 1);
        count_buf[MAX_LINE - 1] = '\0';
        total_cols = split_line(count_buf, delim, fields, MAX_COLS);
    } else {
        strncpy(data_line, first_line_copy, MAX_LINE - 1);
        data_line[MAX_LINE - 1] = '\0';
        total_cols = hcount;
    }

    if (total_cols < 2) {
        LOG_ERROR("[csv] Too few columns: %d", total_cols);
        fclose(f);
        return -1;
    }

    /* Resolve target column */
    int tgt = target_col;
    if (tgt < 0) tgt = total_cols + tgt; /* -1 = last */
    if (tgt < 0 || tgt >= total_cols) {
        LOG_ERROR("[csv] Invalid target column: %d (total: %d)", target_col, total_cols);
        fclose(f);
        return -1;
    }

    int nfeat = total_cols - 1;

    /* First pass: count lines */
    long data_start = has_header ? ftell(f) : 0;
    /* We need to account for the first data line we already read */
    int line_count = 1; /* We already have data_line */
    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) > 1) line_count++;
    }

    /* Allocate */
    float* X = malloc(sizeof(float) * line_count * nfeat);
    float* y = malloc(sizeof(float) * line_count);
    if (!X || !y) { free(X); free(y); fclose(f); return -1; }

    LabelMap lm = {.count = 0};
    int row = 0;

    /* Parse first data line */
    {
        char parse_buf[MAX_LINE];
        strncpy(parse_buf, data_line, MAX_LINE - 1);
        parse_buf[MAX_LINE - 1] = '\0';
        int nc = split_line(parse_buf, delim, fields, MAX_COLS);
        if (nc >= total_cols) {
            int fi = 0;
            for (int j = 0; j < total_cols; j++) {
                if (j == tgt) {
                    if (is_numeric(fields[j]))
                        y[row] = (float)strtod(fields[j], NULL);
                    else
                        y[row] = (float)label_map_get_or_add(&lm, fields[j]);
                } else {
                    X[row * nfeat + fi] = (float)strtod(fields[j], NULL);
                    fi++;
                }
            }
            row++;
        }
    }

    /* Seek to after header if needed, then parse rest */
    if (has_header) {
        fseek(f, data_start, SEEK_SET);
        /* Skip the first data line we already processed */
        fgets(line, sizeof(line), f);
    } else {
        rewind(f);
        /* Skip the first data line */
        fgets(line, sizeof(line), f);
    }

    while (fgets(line, sizeof(line), f) && row < line_count) {
        if (strlen(line) < 2) continue;
        char parse_buf[MAX_LINE];
        strncpy(parse_buf, line, MAX_LINE - 1);
        parse_buf[MAX_LINE - 1] = '\0';

        int nc = split_line(parse_buf, delim, fields, MAX_COLS);
        if (nc < total_cols) continue;

        int fi = 0;
        for (int j = 0; j < total_cols; j++) {
            if (j == tgt) {
                if (is_numeric(fields[j]))
                    y[row] = (float)strtod(fields[j], NULL);
                else
                    y[row] = (float)label_map_get_or_add(&lm, fields[j]);
            } else {
                X[row * nfeat + fi] = (float)strtod(fields[j], NULL);
                fi++;
            }
        }
        row++;
    }
    fclose(f);

    /* Output */
    *X_out = X;
    *y_out = y;
    *num_samples = row;
    *num_features = nfeat;
    *num_classes = lm.count > 0 ? lm.count : 0;

    if (class_names_out && lm.count > 0) {
        char** names = malloc(sizeof(char*) * lm.count);
        for (int i = 0; i < lm.count; i++) {
            names[i] = strdup(lm.labels[i]);
        }
        *class_names_out = names;
    } else if (class_names_out) {
        *class_names_out = NULL;
    }

    LOG_INFO("[csv] Parsed %s: %d samples, %d features, %d classes",
             filepath, row, nfeat, lm.count);
    return 0;
}

Dataset* cml_dataset_from_csv(const char* filepath, int target_col) {
    if (!filepath) return NULL;

    float* X = NULL; float* y = NULL;
    int n = 0, nf = 0, nc = 0;
    char** class_names = NULL;

    if (cml_csv_parse(filepath, target_col, &X, &y, &n, &nf, &nc, &class_names) != 0)
        return NULL;

    Dataset* ds = dataset_from_arrays(X, y, n, nf, 1);
    if (ds) {
        ds->num_classes = nc;
        ds->class_names = class_names;
        cml_dataset_compute_stats(ds);
    }
    free(X); free(y);
    return ds;
}
