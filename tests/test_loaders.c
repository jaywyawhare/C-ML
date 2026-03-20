#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "cml.h"
#include "datasets/loaders.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    printf("  Testing: %s... ", #name); \
    tests_run++; \
    if (test_##name()) { \
        printf("PASS\n"); \
        tests_passed++; \
    } else { \
        printf("FAIL\n"); \
    } \
} while(0)

static void mkdirs(const char* path) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "mkdir -p '%s'", path);
    system(cmd);
}

static void write_ppm(const char* path, int w, int h, unsigned char r, unsigned char g, unsigned char b) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", w, h);
    for (int i = 0; i < w * h; i++) {
        fputc(r, f); fputc(g, f); fputc(b, f);
    }
    fclose(f);
}

static char tmpdir[256];

static void setup_image_dir(void) {
    snprintf(tmpdir, sizeof(tmpdir), "/tmp/cml_test_loaders_%d", getpid());
    char cls0[512], cls1[512];
    snprintf(cls0, sizeof(cls0), "%s/images/cat", tmpdir);
    snprintf(cls1, sizeof(cls1), "%s/images/dog", tmpdir);
    mkdirs(cls0);
    mkdirs(cls1);

    char path[512];
    snprintf(path, sizeof(path), "%s/img0.ppm", cls0);
    write_ppm(path, 8, 8, 255, 0, 0);
    snprintf(path, sizeof(path), "%s/img1.ppm", cls0);
    write_ppm(path, 8, 8, 200, 0, 0);
    snprintf(path, sizeof(path), "%s/img0.ppm", cls1);
    write_ppm(path, 8, 8, 0, 0, 255);
}

static void setup_squad_file(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/squad.json", tmpdir);
    mkdirs(tmpdir);
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f,
        "{\n"
        "  \"data\": [{\n"
        "    \"title\": \"Test\",\n"
        "    \"paragraphs\": [{\n"
        "      \"context\": \"The quick brown fox jumps over the lazy dog.\",\n"
        "      \"qas\": [{\n"
        "        \"question\": \"What color is the fox?\",\n"
        "        \"id\": \"q1\",\n"
        "        \"answers\": [{\n"
        "          \"text\": \"brown\",\n"
        "          \"answer_start\": 10\n"
        "        }]\n"
        "      }]\n"
        "    }]\n"
        "  }]\n"
        "}\n");
    fclose(f);
}

static void setup_librispeech_dir(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/audio/1-2", tmpdir);
    mkdirs(dir);

    char path[512];
    snprintf(path, sizeof(path), "%s/1-2-0001.flac", dir);
    FILE* f = fopen(path, "w");
    if (f) { fprintf(f, "fake_audio"); fclose(f); }

    snprintf(path, sizeof(path), "%s/1-2.trans.txt", dir);
    f = fopen(path, "w");
    if (f) { fprintf(f, "1-2-0001 HELLO WORLD\n"); fclose(f); }
}

static void cleanup(void) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", tmpdir);
    system(cmd);
}

static int test_imagenet_open(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/images", tmpdir);

    CMLImageNetLoader* loader = cml_imagenet_open(dir, 16);
    if (!loader) return 0;
    if (loader->num_samples != 3) { cml_imagenet_free(loader); return 0; }
    if (loader->num_classes != 2) { cml_imagenet_free(loader); return 0; }
    if (loader->image_size != 16) { cml_imagenet_free(loader); return 0; }
    cml_imagenet_free(loader);
    return 1;
}

static int test_imagenet_load_batch(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/images", tmpdir);

    CMLImageNetLoader* loader = cml_imagenet_open(dir, 4);
    if (!loader) return 0;

    Dataset* ds = cml_imagenet_load_batch(loader, 0, 2);
    if (!ds) { cml_imagenet_free(loader); return 0; }
    if (ds->num_samples != 2) { dataset_free(ds); cml_imagenet_free(loader); return 0; }
    if (ds->input_size != 3 * 4 * 4) { dataset_free(ds); cml_imagenet_free(loader); return 0; }

    dataset_free(ds);
    cml_imagenet_free(loader);
    return 1;
}

static int test_load_image_folder(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/images", tmpdir);

    Dataset* ds = cml_load_image_folder(dir, 4);
    if (!ds) return 0;
    if (ds->num_samples != 3) { dataset_free(ds); return 0; }
    dataset_free(ds);
    return 1;
}

static int test_squad_open(void) {
    char path[512];
    snprintf(path, sizeof(path), "%s/squad.json", tmpdir);

    CMLSQuADLoader* loader = cml_squad_open(path);
    if (!loader) return 0;
    if (loader->num_samples != 1) { cml_squad_free(loader); return 0; }
    if (strcmp(loader->questions[0], "What color is the fox?") != 0) { cml_squad_free(loader); return 0; }
    if (strcmp(loader->answers[0], "brown") != 0) { cml_squad_free(loader); return 0; }
    if (loader->answer_starts[0] != 10) { cml_squad_free(loader); return 0; }
    cml_squad_free(loader);
    return 1;
}

static int test_librispeech_open(void) {
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/audio", tmpdir);

    CMLLibriSpeechLoader* loader = cml_librispeech_open(dir);
    if (!loader) return 0;
    if (loader->num_samples != 1) { cml_librispeech_free(loader); return 0; }
    if (strcmp(loader->transcripts[0], "HELLO WORLD") != 0) { cml_librispeech_free(loader); return 0; }
    if (loader->sample_rate != 16000) { cml_librispeech_free(loader); return 0; }
    cml_librispeech_free(loader);
    return 1;
}

static int test_imagenet_null_safety(void) {
    if (cml_imagenet_open(NULL, 224) != NULL) return 0;
    if (cml_imagenet_load_batch(NULL, 0, 1) != NULL) return 0;
    cml_imagenet_free(NULL);
    return 1;
}

static int test_squad_null_safety(void) {
    if (cml_squad_open(NULL) != NULL) return 0;
    if (cml_squad_open("/nonexistent/path.json") != NULL) return 0;
    cml_squad_free(NULL);
    return 1;
}

int main(void) {
    printf("=== Dataset Loader Tests ===\n");

    setup_image_dir();
    setup_squad_file();
    setup_librispeech_dir();

    TEST(imagenet_open);
    TEST(imagenet_load_batch);
    TEST(load_image_folder);
    TEST(squad_open);
    TEST(librispeech_open);
    TEST(imagenet_null_safety);
    TEST(squad_null_safety);

    cleanup();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
