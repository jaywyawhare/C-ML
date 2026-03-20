#ifndef CML_NN_OPENAI_API_H
#define CML_NN_OPENAI_API_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLOpenAIServer {
    int port;
    int listen_fd;
    bool running;
    void* model;
    void* tokenizer;
    int max_tokens;
    float temperature;
    float top_p;
    char model_name[128];
    char model_path[512];
} CMLOpenAIServer;

CMLOpenAIServer* cml_openai_server_create(int port);
int cml_openai_server_load_model(CMLOpenAIServer* srv, const char* model_path);
int cml_openai_server_run(CMLOpenAIServer* srv);
void cml_openai_server_stop(CMLOpenAIServer* srv);
void cml_openai_server_free(CMLOpenAIServer* srv);

#ifdef __cplusplus
}
#endif

#endif /* CML_NN_OPENAI_API_H */
