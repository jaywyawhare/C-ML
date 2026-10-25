#define _POSIX_C_SOURCE 200809L
#include "datasets/datasets.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "alloc/cml_allocator.h"

#define MAX_LINE 8192
#define MAX_COLS 256
#define MAX_LABELS 64
#define MAX_NAME 96

#define TWO_PI 6.283185307179586

/**
 * Return non-zero if the string parses as a number, tolerating leading spaces,
 * sign, decimal point, and exponent notation. Used to tell data rows from headers.
 * @param s (const char*)
 * @return (int)
 */
static int is_numeric(const char* s) {
    if (!s || !*s)
        return 0;
    while (*s == ' ')
        s++;
    if (*s == '-' || *s == '+')
        s++;
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

/**
 * Skip spaces and tabs, returning the first non-blank character.
 * @param s (const char*)
 * @return (const char*)
 */
static const char* skip_blanks(const char* s) {
    while (*s == ' ' || *s == '\t')
        s++;
    return s;
}

/**
 * Read an integer of 1..maxlen digits; store it and advance *s past the digits.
 * @param s (const char**)
 * @param maxlen (int)
 * @param out (int*)
 * @return (int)
 */
static int read_int(const char** s, int maxlen, int* out) {
    const char* p = *s;
    int n = 0, digits = 0;
    while (digits < maxlen && isdigit((unsigned char)*p)) {
        n = n * 10 + (*p - '0');
        p++;
        digits++;
    }
    if (digits == 0)
        return 0;
    *out = n;
    *s   = p;
    return 1;
}

/**
 * Parse a calendar date at *s. ISO `YYYY-MM-DD` is always accepted; `/` or `.`
 * separated dates are read in @p order (day/month/year or month/day/year), with a
 * two-digit year taken as 20xx. On success advances *s past the date and returns 1.
 * @param s (const char**)
 * @param order (CSVDateOrder)
 * @param y (int*)
 * @param mo (int*)
 * @param d (int*)
 * @return (int)
 */
static int parse_date(const char** s, CSVDateOrder order, int* y, int* mo, int* d) {
    const char* p = *s;
    int a;
    if (!read_int(&p, 4, &a))
        return 0;
    char sep = *p;
    if (sep != '-' && sep != '/' && sep != '.')
        return 0;
    p++;
    int b;
    if (!read_int(&p, 2, &b) || *p != sep)
        return 0;
    p++;
    int c;
    if (!read_int(&p, 4, &c))
        return 0;

    int yy, mm, dd;
    if (sep == '-') {
        yy = a;
        mm = b;
        dd = c; /* ISO 8601, accepted in any order */
    } else if (order == CML_CSV_DATE_MDY) {
        mm = a;
        dd = b;
        yy = c;
    } else if (order == CML_CSV_DATE_DMY) {
        dd = a;
        mm = b;
        yy = c;
    } else {
        return 0; /* ISO order: only dash-separated dates are auto-parsed */
    }
    if (yy < 100)
        yy += 2000;
    if (mm < 1 || mm > 12 || dd < 1 || dd > 31 || yy < 1 || yy > 9999)
        return 0;

    *y  = yy;
    *mo = mm;
    *d  = dd;
    *s  = p;
    return 1;
}

/**
 * Parse `HH:MM` with optional `:SS` and `.fff`; advance *s and return 1 on success.
 * @param s (const char**)
 * @param h (int*)
 * @param mi (int*)
 * @param se (int*)
 * @return (int)
 */
static int parse_clock(const char** s, int* h, int* mi, int* se) {
    const char* p = *s;
    int hh, mm, ss = 0;
    if (!read_int(&p, 2, &hh) || *p != ':')
        return 0;
    p++;
    if (!read_int(&p, 2, &mm))
        return 0;
    if (*p == ':') {
        p++;
        if (!read_int(&p, 2, &ss))
            return 0;
        if (*p == '.') {
            p++;
            int frac;
            read_int(&p, 9, &frac); /* fractional seconds are consumed but dropped */
        }
    }
    if (hh > 23 || mm > 59 || ss > 60)
        return 0;
    *h  = hh;
    *mi = mm;
    *se = ss;
    *s  = p;
    return 1;
}

/**
 * Day of week by Sakamoto's method: 0 = Sunday .. 6 = Saturday.
 * @param y (int)
 * @param m (int)
 * @param d (int)
 * @return (int)
 */
static int day_of_week(int y, int m, int d) {
    static const int t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    if (m < 3)
        y -= 1;
    return (y + y / 4 - y / 100 + y / 400 + t[m - 1] + d) % 7;
}

/** How a column's cells are turned into numeric features. */
typedef enum {
    COL_NUMERIC,
    COL_CATEGORICAL,
    COL_DATE,
    COL_TIME,
    COL_DATETIME,
} ColKind;

/**
 * Classify a trimmed cell: a date and clock separated by `T`/space is COL_DATETIME,
 * either alone is COL_DATE/COL_TIME, a plain number is COL_NUMERIC, anything else
 * (text, a label) is COL_CATEGORICAL.
 * @param raw (const char*)
 * @param order (CSVDateOrder)
 * @return (ColKind)
 */
static ColKind classify_cell(const char* raw, CSVDateOrder order) {
    const char* s = skip_blanks(raw);
    int y, mo, d, h, mi, se;
    const char* p = s;
    if (parse_date(&p, order, &y, &mo, &d)) {
        if (*p == 'T' || *p == ' ') {
            const char* q = p + 1;
            if (parse_clock(&q, &h, &mi, &se) && *skip_blanks(q) == '\0')
                return COL_DATETIME;
        }
        if (*skip_blanks(p) == '\0')
            return COL_DATE;
    }
    p = s;
    if (parse_clock(&p, &h, &mi, &se) && *skip_blanks(p) == '\0')
        return COL_TIME;
    if (is_numeric(raw))
        return COL_NUMERIC;
    return COL_CATEGORICAL;
}

/**
 * Heap-allocate "base.suffix", or a copy of base when suffix is empty.
 * @param base (const char*)
 * @param suffix (const char*)
 * @return (char*)
 */
static char* make_feat_name(const char* base, const char* suffix) {
    if (!base)
        return NULL;
    if (!suffix || !*suffix)
        return cml_strdup(base);
    size_t n = strlen(base) + 1 + strlen(suffix) + 1;
    char* s  = cml_malloc(n);
    if (s)
        snprintf(s, n, "%s.%s", base, suffix);
    return s;
}

/**
 * Emit a date's components (year, month, day, day-of-week, plus month/dow sin-cos
 * when cyclical). Writes values to @p x and names to @p nm when non-NULL; returns
 * the component count, so a NULL/NULL call yields the column width.
 * @param y (int)
 * @param mo (int)
 * @param d (int)
 * @param o (const CSVParseOptions*)
 * @param base (const char*)
 * @param x (float*)
 * @param nm (char**)
 * @return (int)
 */
static int emit_date(int y, int mo, int d, const CSVParseOptions* o, const char* base, float* x,
                     char** nm) {
    int i = 0;
#define PUT(v, suf)                                                                                \
    do {                                                                                           \
        if (x)                                                                                     \
            x[i] = (float)(v);                                                                     \
        if (nm)                                                                                    \
            nm[i] = make_feat_name(base, suf);                                                     \
        i++;                                                                                       \
    } while (0)
    PUT(y, "year");
    PUT(mo, "month");
    PUT(d, "day");
    PUT(day_of_week(y, mo, d), "dow");
    if (o->cyclical) {
        double ma = TWO_PI * ((mo - 1) / 12.0);
        double wa = TWO_PI * (day_of_week(y, mo, d) / 7.0);
        PUT(sin(ma), "month_sin");
        PUT(cos(ma), "month_cos");
        PUT(sin(wa), "dow_sin");
        PUT(cos(wa), "dow_cos");
    }
#undef PUT
    return i;
}

/**
 * Emit a clock's components (hour, minute, second, plus hour sin-cos when cyclical).
 * Writes values to @p x and names to @p nm when non-NULL; returns the component count.
 * @param h (int)
 * @param mi (int)
 * @param se (int)
 * @param o (const CSVParseOptions*)
 * @param base (const char*)
 * @param x (float*)
 * @param nm (char**)
 * @return (int)
 */
static int emit_time(int h, int mi, int se, const CSVParseOptions* o, const char* base, float* x,
                     char** nm) {
    int i = 0;
#define PUT(v, suf)                                                                                \
    do {                                                                                           \
        if (x)                                                                                     \
            x[i] = (float)(v);                                                                     \
        if (nm)                                                                                    \
            nm[i] = make_feat_name(base, suf);                                                     \
        i++;                                                                                       \
    } while (0)
    PUT(h, "hour");
    PUT(mi, "minute");
    PUT(se, "second");
    if (o->cyclical) {
        double ha = TWO_PI * (h / 24.0);
        PUT(sin(ha), "hour_sin");
        PUT(cos(ha), "hour_cos");
    }
#undef PUT
    return i;
}

/**
 * Number of feature columns a source column of @p kind expands into.
 * @param kind (ColKind)
 * @param o (const CSVParseOptions*)
 * @return (int)
 */
static int col_width(ColKind kind, const CSVParseOptions* o) {
    switch (kind) {
    case COL_DATE:
        return emit_date(2000, 1, 1, o, NULL, NULL, NULL);
    case COL_TIME:
        return emit_time(0, 0, 0, o, NULL, NULL, NULL);
    case COL_DATETIME:
        return emit_date(2000, 1, 1, o, NULL, NULL, NULL) + emit_time(0, 0, 0, o, NULL, NULL, NULL);
    default:
        return 1;
    }
}

/**
 * Write col_width(kind) names for @p base into nm (each heap-allocated).
 * @param kind (ColKind)
 * @param o (const CSVParseOptions*)
 * @param base (const char*)
 * @param nm (char**)
 * @return (void)
 */
static void col_names(ColKind kind, const CSVParseOptions* o, const char* base, char** nm) {
    switch (kind) {
    case COL_DATE:
        emit_date(2000, 1, 1, o, base, NULL, nm);
        break;
    case COL_TIME:
        emit_time(0, 0, 0, o, base, NULL, nm);
        break;
    case COL_DATETIME: {
        int dw = emit_date(2000, 1, 1, o, base, NULL, nm);
        emit_time(0, 0, 0, o, base, NULL, nm + dw);
        break;
    }
    default:
        nm[0] = make_feat_name(base, NULL);
        break;
    }
}

/**
 * Parse a date/time cell into its numeric components, writing col_width(kind) floats
 * to @p x. A cell that fails to parse (e.g. a stray row under a column typed from the
 * first data row) is written as zeros so the feature matrix stays rectangular.
 * @param raw (const char*)
 * @param kind (ColKind)
 * @param o (const CSVParseOptions*)
 * @param x (float*)
 * @return (void)
 */
static void fill_datetime(const char* raw, ColKind kind, const CSVParseOptions* o, float* x) {
    const char* s = skip_blanks(raw);
    int y = 0, mo = 1, d = 1, h = 0, mi = 0, se = 0;
    int ok = 0;
    if (kind == COL_TIME) {
        const char* p = s;
        ok            = parse_clock(&p, &h, &mi, &se);
        if (ok)
            emit_time(h, mi, se, o, NULL, x, NULL);
    } else {
        const char* p = s;
        if (parse_date(&p, o->date_order, &y, &mo, &d)) {
            int dw = emit_date(y, mo, d, o, NULL, x, NULL);
            if (kind == COL_DATETIME) {
                if (*p == 'T' || *p == ' ') {
                    const char* q = p + 1;
                    parse_clock(&q, &h, &mi, &se);
                }
                emit_time(h, mi, se, o, NULL, x + dw, NULL);
            }
            ok = 1;
        }
    }
    if (!ok)
        memset(x, 0, (size_t)col_width(kind, o) * sizeof(float));
}

/**
 * Guess the field delimiter of a line by majority count: tab, then semicolon,
 * otherwise defaulting to comma.
 * @param line (const char*)
 * @return (char)
 */
static char detect_delimiter(const char* line) {
    int commas = 0, semicolons = 0, tabs = 0;
    for (const char* p = line; *p; p++) {
        if (*p == ',')
            commas++;
        else if (*p == ';')
            semicolons++;
        else if (*p == '\t')
            tabs++;
    }
    if (tabs > commas && tabs > semicolons)
        return '\t';
    if (semicolons > commas)
        return ';';
    return ',';
}

/**
 * Split a line by delimiter, return field count. Fields written into fields[]
 * @param line (char*)
 * @param delim (char)
 * @param fields (char**)
 * @param max_fields (int)
 * @return (int)
 */
static int split_line(char* line, char delim, char** fields, int max_fields) {
    int count = 0;
    char* p   = line;

    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        line[--len] = '\0';

    if (len == 0)
        return 0;

    while (p && count < max_fields) {
        fields[count++] = p;
        char* next      = strchr(p, delim);
        if (next) {
            *next = '\0';
            p     = next + 1;
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

/**
 * Return the index of a (whitespace-trimmed) label, inserting it if new. Returns
 * -1 on an empty label or when the fixed MAX_LABELS table is full.
 * @param lm (LabelMap*)
 * @param s (const char*)
 * @return (int)
 */
static int label_map_get_or_add(LabelMap* lm, const char* s) {
    while (*s == ' ')
        s++;
    char trimmed[128];
    strncpy(trimmed, s, 127);
    trimmed[127] = '\0';
    size_t tlen  = strlen(trimmed);
    while (tlen > 0 &&
           (trimmed[tlen - 1] == ' ' || trimmed[tlen - 1] == '\r' || trimmed[tlen - 1] == '\n'))
        trimmed[--tlen] = '\0';

    if (tlen == 0)
        return -1;

    for (int i = 0; i < lm->count; i++) {
        if (strcmp(lm->labels[i], trimmed) == 0)
            return i;
    }
    if (lm->count >= MAX_LABELS)
        return -1;
    strcpy(lm->labels[lm->count], trimmed);
    return lm->count++;
}

/**
 * Default parse options: last column is the target, auto header/delimiter, ISO
 * dates, date/time columns expanded into numeric components, no cyclical encoding.
 * @return (CSVParseOptions)
 */
CSVParseOptions cml_csv_default_options(void) {
    CSVParseOptions o = {
        .target_col      = -1,
        .header          = -1,
        .delimiter       = 0,
        .date_order      = CML_CSV_DATE_ISO,
        .expand_datetime = 1,
        .cyclical        = 0,
    };
    return o;
}

int cml_csv_parse_ex(const char* filepath, const CSVParseOptions* opts, float** X_out,
                     float** y_out, int* num_samples, int* num_features, int* num_classes,
                     char*** class_names_out, char*** feature_names_out) {
    CSVParseOptions o = opts ? *opts : cml_csv_default_options();

    FILE* f = fopen(filepath, "r");
    if (!f) {
        LOG_ERROR("[csv] Cannot open: %s", filepath);
        return -1;
    }

    char line[MAX_LINE];
    char* fields[MAX_COLS];

    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return -1;
    }

    char delim = o.delimiter ? o.delimiter : detect_delimiter(line);

    /* First line decides the header: forced by opts, else inferred from whether its
     * first field is a label rather than a number or date. */
    char first_buf[MAX_LINE];
    snprintf(first_buf, sizeof(first_buf), "%s", line);
    int fcount = split_line(first_buf, delim, fields, MAX_COLS);
    int has_header;
    if (o.header >= 0)
        has_header = o.header;
    else
        has_header = (fcount > 0 && classify_cell(fields[0], o.date_order) == COL_CATEGORICAL);

    /* Column names come from the header, else col0/col1/... */
    char colnames[MAX_COLS][MAX_NAME];
    if (has_header) {
        for (int j = 0; j < fcount && j < MAX_COLS; j++) {
            const char* nm = skip_blanks(fields[j]);
            snprintf(colnames[j], MAX_NAME, "%s", nm);
        }
    }

    long data_offset = has_header ? ftell(f) : 0;

    char data_line[MAX_LINE];
    if (fseek(f, data_offset, SEEK_SET) != 0 || !fgets(data_line, sizeof(data_line), f)) {
        fclose(f);
        return -1; /* no data rows */
    }
    char classify_buf[MAX_LINE];
    snprintf(classify_buf, sizeof(classify_buf), "%s", data_line);
    int total_cols = split_line(classify_buf, delim, fields, MAX_COLS);

    if (total_cols < 2) {
        LOG_ERROR("[csv] Too few columns: %d", total_cols);
        fclose(f);
        return -1;
    }
    /* Name any columns the header did not cover (or all of them when headerless). */
    for (int j = has_header ? fcount : 0; j < total_cols; j++)
        snprintf(colnames[j], MAX_NAME, "col%d", j);

    int tgt = o.target_col;
    if (tgt < 0)
        tgt = total_cols + tgt;
    if (tgt < 0 || tgt >= total_cols) {
        LOG_ERROR("[csv] Invalid target column: %d (total: %d)", o.target_col, total_cols);
        fclose(f);
        return -1;
    }

    /* Type each column from the first data row, and lay out the expanded features. */
    ColKind kind[MAX_COLS];
    int offset[MAX_COLS];
    int nfeat = 0;
    for (int j = 0; j < total_cols; j++) {
        ColKind k = classify_cell(fields[j], o.date_order);
        if (!o.expand_datetime && (k == COL_DATE || k == COL_TIME || k == COL_DATETIME))
            k = COL_NUMERIC;
        kind[j] = k;
        if (j == tgt)
            continue;
        offset[j] = nfeat;
        nfeat += col_width(k, &o);
    }

    LabelMap* colmaps = cml_calloc((size_t)total_cols, sizeof(LabelMap));
    if (!colmaps) {
        fclose(f);
        return -1;
    }

    /* Count data rows from the same anchor used to parse them. */
    if (fseek(f, data_offset, SEEK_SET) != 0) {
        cml_free(colmaps);
        fclose(f);
        return -1;
    }
    int line_count = 0;
    while (fgets(line, sizeof(line), f))
        if (strlen(line) > 1)
            line_count++;

    float* X          = cml_malloc(sizeof(float) * (size_t)line_count * (size_t)nfeat);
    float* y          = cml_malloc(sizeof(float) * (size_t)line_count);
    char** feat_names = feature_names_out ? cml_calloc((size_t)nfeat, sizeof(char*)) : NULL;
    if (!X || !y || (feature_names_out && !feat_names)) {
        cml_free(X);
        cml_free(y);
        cml_free(feat_names);
        cml_free(colmaps);
        fclose(f);
        return -1;
    }

    if (feat_names)
        for (int j = 0; j < total_cols; j++)
            if (j != tgt)
                col_names(kind[j], &o, colnames[j], feat_names + offset[j]);

    if (fseek(f, data_offset, SEEK_SET) != 0) {
        cml_free(X);
        cml_free(y);
        cml_free(feat_names);
        cml_free(colmaps);
        fclose(f);
        return -1;
    }

    int row = 0;
    while (fgets(line, sizeof(line), f) && row < line_count) {
        if (strlen(line) < 2)
            continue;
        char parse_buf[MAX_LINE];
        strncpy(parse_buf, line, MAX_LINE - 1);
        parse_buf[MAX_LINE - 1] = '\0';

        int nc = split_line(parse_buf, delim, fields, MAX_COLS);
        if (nc < total_cols)
            continue;

        for (int j = 0; j < total_cols; j++) {
            if (j == tgt) {
                if (kind[j] == COL_NUMERIC)
                    y[row] = (float)strtod(fields[j], NULL);
                else
                    y[row] = (float)label_map_get_or_add(&colmaps[j], fields[j]);
                continue;
            }
            float* cell = X + (size_t)row * nfeat + offset[j];
            switch (kind[j]) {
            case COL_NUMERIC:
                cell[0] = (float)strtod(fields[j], NULL);
                break;
            case COL_CATEGORICAL:
                cell[0] = (float)label_map_get_or_add(&colmaps[j], fields[j]);
                break;
            default:
                fill_datetime(fields[j], kind[j], &o, cell);
                break;
            }
        }
        row++;
    }
    fclose(f);

    LabelMap* tgtmap = &colmaps[tgt];
    int nclasses     = tgtmap->count;
    *X_out           = X;
    *y_out           = y;
    *num_samples     = row;
    *num_features    = nfeat;
    *num_classes     = nclasses;

    if (class_names_out) {
        char** names = nclasses > 0 ? cml_malloc(sizeof(char*) * nclasses) : NULL;
        if (names)
            for (int i = 0; i < nclasses; i++)
                names[i] = cml_strdup(tgtmap->labels[i]);
        *class_names_out = names;
    }
    if (feature_names_out)
        *feature_names_out = feat_names;

    cml_free(colmaps);
    LOG_INFO("[csv] Parsed %s: %d samples, %d features, %d classes", filepath, row, nfeat,
             nclasses);
    return 0;
}

/**
 * Parse a delimited file into feature/target arrays, auto-detecting the delimiter
 * and an optional header row. target_col may be negative to index from the end;
 * date/time columns are split into numeric components and non-numeric targets are
 * mapped to class indices. Allocates *X_out, *y_out, and optionally *class_names_out
 * (all cml_malloc, caller frees). Returns 0 on success.
 * @param filepath (const char*)
 * @param target_col (int)
 * @param X_out (float**)
 * @param y_out (float**)
 * @param num_samples (int*)
 * @param num_features (int*)
 * @param num_classes (int*)
 * @param class_names_out (char***)
 * @return (int)
 */
int cml_csv_parse(const char* filepath, int target_col, float** X_out, float** y_out,
                  int* num_samples, int* num_features, int* num_classes, char*** class_names_out) {
    CSVParseOptions o = cml_csv_default_options();
    o.target_col      = target_col;
    return cml_csv_parse_ex(filepath, &o, X_out, y_out, num_samples, num_features, num_classes,
                            class_names_out, NULL);
}

/**
 * Load a CSV file into a Dataset, using target_col as the label column, and
 * populate feature names and statistics. Returns NULL on error; the caller owns
 * the Dataset.
 * @param filepath (const char*)
 * @param target_col (int)
 * @return (Dataset*)
 */
Dataset* cml_dataset_from_csv(const char* filepath, int target_col) {
    if (!filepath)
        return NULL;

    CSVParseOptions o = cml_csv_default_options();
    o.target_col      = target_col;

    float* X = NULL;
    float* y = NULL;
    int n = 0, nf = 0, nc = 0;
    char** class_names   = NULL;
    char** feature_names = NULL;

    if (cml_csv_parse_ex(filepath, &o, &X, &y, &n, &nf, &nc, &class_names, &feature_names) != 0)
        return NULL;

    Dataset* ds = dataset_from_arrays(X, y, n, nf, 1);
    if (ds) {
        ds->num_classes   = nc;
        ds->class_names   = class_names;
        ds->feature_names = feature_names;
        cml_dataset_compute_stats(ds);
    } else {
        if (class_names) {
            for (int i = 0; i < nc; i++)
                cml_free(class_names[i]);
            cml_free(class_names);
        }
        if (feature_names) {
            for (int i = 0; i < nf; i++)
                cml_free(feature_names[i]);
            cml_free(feature_names);
        }
    }
    cml_free(X);
    cml_free(y);
    return ds;
}
