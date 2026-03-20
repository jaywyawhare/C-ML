#ifndef CML_DATASET_LOADERS_H
#define CML_DATASET_LOADERS_H

#include "datasets/datasets.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLImageNetLoader {
    char** image_paths;
    int* labels;
    int num_samples;
    int num_classes;
    int image_size;
} CMLImageNetLoader;

CMLImageNetLoader* cml_imagenet_open(const char* dir_path, int image_size);
Dataset* cml_imagenet_load_batch(CMLImageNetLoader* loader, int offset, int batch_size);
void cml_imagenet_free(CMLImageNetLoader* loader);

typedef struct CMLLibriSpeechLoader {
    char** audio_paths;
    char** transcripts;
    int num_samples;
    int sample_rate;
} CMLLibriSpeechLoader;

CMLLibriSpeechLoader* cml_librispeech_open(const char* dir_path);
void cml_librispeech_free(CMLLibriSpeechLoader* loader);

typedef struct CMLSQuADLoader {
    char** contexts;
    char** questions;
    char** answers;
    int* answer_starts;
    int num_samples;
} CMLSQuADLoader;

CMLSQuADLoader* cml_squad_open(const char* json_path);
void cml_squad_free(CMLSQuADLoader* loader);

Dataset* cml_load_image_folder(const char* dir_path, int image_size);

typedef struct CMLKiTS19Loader {
    char** case_dirs;
    int num_cases;
} CMLKiTS19Loader;

CMLKiTS19Loader* cml_kits19_open(const char* data_dir);
void cml_kits19_free(CMLKiTS19Loader* loader);
int cml_kits19_load_case(CMLKiTS19Loader* loader, int case_idx,
                         Tensor** volume, Tensor** segmentation);

typedef struct CMLOpenImagesLoader {
    char** image_ids;
    int num_images;
    char* images_dir;
    char* annotations_path;
} CMLOpenImagesLoader;

CMLOpenImagesLoader* cml_openimages_open(const char* images_dir, const char* annotations_csv);
void cml_openimages_free(CMLOpenImagesLoader* loader);

typedef struct CMLWikipediaLoader {
    char** article_paths;
    int num_articles;
    size_t total_bytes;
} CMLWikipediaLoader;

CMLWikipediaLoader* cml_wikipedia_open(const char* dump_dir);
void cml_wikipedia_free(CMLWikipediaLoader* loader);
int cml_wikipedia_read_chunk(CMLWikipediaLoader* loader, int article_idx,
                             char* buf, size_t buf_size, size_t* bytes_read);

#ifdef __cplusplus
}
#endif

#endif /* CML_DATASET_LOADERS_H */
