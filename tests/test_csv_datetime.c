/* CSV loader: row/header handling and date/time column expansion.
 *
 * Guards two parser bugs (a dropped second data row under a header, and a
 * headerless first row taken for a header when it starts with a date) and the
 * date/time feature expansion, categorical feature mapping and feature naming.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>
#include "alloc/cml_allocator.h"

#include "cml.h"
#include "core/dataset.h"
#include "datasets/datasets.h"
#include "test_harness.h"

#define TMP_DIR "/tmp/cml_csv_dt_tests"

static void write_csv(const char* name, const char* body) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", TMP_DIR, name);
    FILE* f = fopen(path, "wb");
    if (f) {
        fputs(body, f);
        fclose(f);
    }
}

static const char* csv_path(const char* name) {
    static char path[512];
    snprintf(path, sizeof(path), "%s/%s", TMP_DIR, name);
    return path;
}

/* A header plus three numeric rows must yield three samples: the second row was
 * dropped before the data-offset anchor was fixed. */
static int test_header_keeps_every_row(void) {
    write_csv("num.csv", "x,y,z\n1,2,0\n3,4,1\n5,6,0\n");
    float *X = NULL, *y = NULL;
    int n = 0, nf = 0, nc = 0;
    int rc = cml_csv_parse(csv_path("num.csv"), -1, &X, &y, &n, &nf, &nc, NULL);
    int ok = rc == 0 && n == 3 && nf == 2 && X[0] == 1.0f && X[2 * 2 + 0] == 5.0f && y[2] == 0.0f;
    cml_free(X);
    cml_free(y);
    return ok;
}

/* A headerless file whose first column is an ISO date must keep its first row,
 * rather than mistaking it for a header. */
static int test_leading_date_is_not_a_header(void) {
    write_csv("nohdr.csv", "2024-01-15,3.5,0\n2024-07-04,28.1,1\n");
    float *X = NULL, *y = NULL;
    int n = 0, nf = 0, nc = 0;
    int rc = cml_csv_parse(csv_path("nohdr.csv"), -1, &X, &y, &n, &nf, &nc, NULL);
    /* date -> year,month,day,dow (4) plus the numeric feature = 5 */
    int ok = rc == 0 && n == 2 && nf == 5 && X[0] == 2024.0f && X[1] == 1.0f && X[2] == 15.0f;
    cml_free(X);
    cml_free(y);
    return ok;
}

/* A date column expands to year/month/day/dow with named features, and a text
 * feature column maps to class indices instead of becoming zero. */
static int test_date_and_text_expansion(void) {
    write_csv("mix.csv", "date,temp,city,label\n2024-01-15,3.5,Paris,0\n2024-07-04,28.1,Rome,1\n"
                         "2023-12-31,-2.0,Oslo,0\n");
    float *X = NULL, *y = NULL;
    int n = 0, nf = 0, nc = 0;
    char** names      = NULL;
    CSVParseOptions o = cml_csv_default_options();
    int rc = cml_csv_parse_ex(csv_path("mix.csv"), &o, &X, &y, &n, &nf, &nc, NULL, &names);
    int ok = rc == 0 && n == 3 && nf == 6;
    /* 2024-01-15 is a Monday (dow=1, 0=Sunday); 2023-12-31 a Sunday (dow=0). */
    ok = ok && X[3] == 1.0f && X[2 * 6 + 3] == 0.0f;
    /* three distinct cities -> 0,1,2 in column 5 */
    ok = ok && X[5] == 0.0f && X[6 + 5] == 1.0f && X[2 * 6 + 5] == 2.0f;
    ok = ok && names && strcmp(names[0], "date.year") == 0 && strcmp(names[3], "date.dow") == 0 &&
         strcmp(names[4], "temp") == 0 && strcmp(names[5], "city") == 0;
    cml_free(X);
    cml_free(y);
    if (names) {
        for (int i = 0; i < nf; i++)
            cml_free(names[i]);
        cml_free(names);
    }
    return ok;
}

/* A datetime column splits into date and clock parts; a time-only column into
 * hour/minute/second. */
static int test_datetime_and_time(void) {
    write_csv("dt.csv", "2024-07-04T14:45:30,3.5,1\n2024-01-15 08:30:00,2.0,0\n");
    float *X = NULL, *y = NULL;
    int n = 0, nf = 0, nc = 0;
    int rc  = cml_csv_parse(csv_path("dt.csv"), -1, &X, &y, &n, &nf, &nc, NULL);
    int okd = rc == 0 && n == 2 && nf == 8 && X[4] == 14.0f && X[5] == 45.0f && X[6] == 30.0f;

    write_csv("time.csv", "08:30,100\n23:59:59,200\n");
    float *Xt = NULL, *yt = NULL;
    int nt = 0, nft = 0, nct = 0;
    int rct = cml_csv_parse(csv_path("time.csv"), -1, &Xt, &yt, &nt, &nft, &nct, NULL);
    int okt = rct == 0 && nt == 2 && nft == 3 && Xt[0] == 8.0f && Xt[1] == 30.0f && Xt[2] == 0.0f &&
              Xt[3 + 2] == 59.0f;
    cml_free(X);
    cml_free(y);
    cml_free(Xt);
    cml_free(yt);
    return okd && okt;
}

/* Cyclical encoding adds sin/cos pairs; disabling expansion keeps one column. */
static int test_cyclical_and_disable(void) {
    write_csv("cyc.csv", "2024-01-15,0\n2024-02-15,1\n");
    float *X = NULL, *y = NULL;
    int n = 0, nf = 0, nc = 0;
    CSVParseOptions o = cml_csv_default_options();
    o.cyclical        = 1;
    int rc            = cml_csv_parse_ex(csv_path("cyc.csv"), &o, &X, &y, &n, &nf, &nc, NULL, NULL);
    /* year,month,day,dow + month_sin/cos + dow_sin/cos = 8 */
    int okc = rc == 0 && nf == 8 && fabsf(X[4] - sinf(0.0f)) < 1e-6f && fabsf(X[5] - 1.0f) < 1e-6f;
    cml_free(X);
    cml_free(y);

    float *Xr = NULL, *yr = NULL;
    int nr = 0, nfr = 0, ncr = 0;
    CSVParseOptions o2 = cml_csv_default_options();
    o2.expand_datetime = 0;
    int rc2 = cml_csv_parse_ex(csv_path("cyc.csv"), &o2, &Xr, &yr, &nr, &nfr, &ncr, NULL, NULL);
    int okr = rc2 == 0 && nfr == 1; /* date left as a single (numeric) column */
    cml_free(Xr);
    cml_free(yr);
    return okc && okr;
}

/* A `/`-separated date is auto-parsed only when a non-ISO order is chosen; under
 * the default ISO order it stays a single numeric column. */
static int test_date_order(void) {
    write_csv("slash.csv", "15/01/2024,10\n04/07/2024,20\n");
    float *Xi = NULL, *yi = NULL;
    int ni = 0, nfi = 0, nci = 0;
    int rci = cml_csv_parse(csv_path("slash.csv"), -1, &Xi, &yi, &ni, &nfi, &nci, NULL);
    int oki = rci == 0 && nfi == 1; /* ISO default: slash date not expanded */
    cml_free(Xi);
    cml_free(yi);

    float *Xd = NULL, *yd = NULL;
    int nd = 0, nfd = 0, ncd = 0;
    CSVParseOptions o = cml_csv_default_options();
    o.date_order      = CML_CSV_DATE_DMY;
    int rcd = cml_csv_parse_ex(csv_path("slash.csv"), &o, &Xd, &yd, &nd, &nfd, &ncd, NULL, NULL);
    /* day/month/year: 15/01/2024 -> day 15, month 1, year 2024 */
    int okd = rcd == 0 && nfd == 4 && Xd[0] == 2024.0f && Xd[1] == 1.0f && Xd[2] == 15.0f;
    cml_free(Xd);
    cml_free(yd);
    return oki && okd;
}

int main(void) {
#ifdef _WIN32
    _mkdir(TMP_DIR);
#else
    mkdir(TMP_DIR, 0755);
#endif
    cml_init();
    printf("=== CSV date/time parsing ===\n");
    TEST(header_keeps_every_row);
    TEST(leading_date_is_not_a_header);
    TEST(date_and_text_expansion);
    TEST(datetime_and_time);
    TEST(cyclical_and_disable);
    TEST(date_order);
    cml_cleanup();
    return TEST_SUMMARY();
}
