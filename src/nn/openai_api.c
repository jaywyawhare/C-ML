#include "nn/openai_api.h"
#include "nn/llama.h"
#include "nn/llm_ops.h"
#include "core/logging.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define HTTP_BUF_SIZE  (256 * 1024)
#define MAX_MESSAGES   64

/* Minimal JSON helpers */

static const char* json_find_key(const char* json, const char* key) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char* p = strstr(json, pattern);
    if (!p) return NULL;
    p += strlen(pattern);
    while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

static int json_read_string(const char* p, char* out, size_t out_size) {
    if (!p || *p != '"') return -1;
    p++;
    size_t i = 0;
    while (*p && *p != '"' && i < out_size - 1) {
        if (*p == '\\' && *(p + 1)) {
            p++;
            switch (*p) {
            case 'n': out[i++] = '\n'; break;
            case 't': out[i++] = '\t'; break;
            case '\\': out[i++] = '\\'; break;
            case '"': out[i++] = '"'; break;
            default: out[i++] = *p; break;
            }
        } else {
            out[i++] = *p;
        }
        p++;
    }
    out[i] = '\0';
    return 0;
}

static float json_read_float(const char* p, float def) {
    if (!p) return def;
    char* end;
    float v = strtof(p, &end);
    return (end != p) ? v : def;
}

static int json_read_int(const char* p, int def) {
    if (!p) return def;
    char* end;
    long v = strtol(p, &end, 10);
    return (end != p) ? (int)v : def;
}

static bool json_read_bool(const char* p, bool def) {
    if (!p) return def;
    if (strncmp(p, "true", 4) == 0) return true;
    if (strncmp(p, "false", 5) == 0) return false;
    return def;
}

typedef struct {
    char role[32];
    char content[8192];
} ChatMessage;

static int parse_messages(const char* json, ChatMessage* msgs, int max_msgs) {
    const char* arr = json_find_key(json, "messages");
    if (!arr || *arr != '[') return 0;
    arr++;

    int count = 0;
    while (*arr && count < max_msgs) {
        const char* obj = strchr(arr, '{');
        if (!obj) break;
        const char* obj_end = strchr(obj, '}');
        if (!obj_end) break;

        size_t obj_len = (size_t)(obj_end - obj + 1);
        char buf[16384];
        if (obj_len >= sizeof(buf)) { arr = obj_end + 1; continue; }
        memcpy(buf, obj, obj_len);
        buf[obj_len] = '\0';

        const char* role_p = json_find_key(buf, "role");
        const char* content_p = json_find_key(buf, "content");

        json_read_string(role_p, msgs[count].role, sizeof(msgs[count].role));
        json_read_string(content_p, msgs[count].content, sizeof(msgs[count].content));
        count++;

        arr = obj_end + 1;
    }
    return count;
}

/* HTTP helpers */

static int recv_http_request(int fd, char* buf, size_t buf_size) {
    size_t total = 0;
    while (total < buf_size - 1) {
        ssize_t n = recv(fd, buf + total, buf_size - 1 - total, 0);
        if (n <= 0) return (total > 0) ? (int)total : -1;
        total += (size_t)n;
        buf[total] = '\0';
        if (strstr(buf, "\r\n\r\n")) {
            char* cl = strstr(buf, "Content-Length:");
            if (!cl) cl = strstr(buf, "content-length:");
            if (cl) {
                int content_len = atoi(cl + 15);
                char* body = strstr(buf, "\r\n\r\n") + 4;
                size_t body_received = total - (size_t)(body - buf);
                while ((int)body_received < content_len && total < buf_size - 1) {
                    n = recv(fd, buf + total, buf_size - 1 - total, 0);
                    if (n <= 0) break;
                    total += (size_t)n;
                    body_received += (size_t)n;
                }
                buf[total] = '\0';
            }
            return (int)total;
        }
    }
    return (int)total;
}

static void send_http_response(int fd, int status_code, const char* status_text,
                               const char* content_type, const char* body) {
    char header[1024];
    size_t body_len = body ? strlen(body) : 0;
    int hdr_len = snprintf(header, sizeof(header),
                           "HTTP/1.1 %d %s\r\n"
                           "Content-Type: %s\r\n"
                           "Content-Length: %zu\r\n"
                           "Access-Control-Allow-Origin: *\r\n"
                           "Connection: close\r\n"
                           "\r\n",
                           status_code, status_text, content_type, body_len);
    send(fd, header, (size_t)hdr_len, MSG_NOSIGNAL);
    if (body && body_len > 0) {
        send(fd, body, body_len, MSG_NOSIGNAL);
    }
}

static void send_sse_chunk(int fd, const char* data) {
    char buf[16384];
    int n = snprintf(buf, sizeof(buf), "data: %s\n\n", data);
    send(fd, buf, (size_t)n, MSG_NOSIGNAL);
}

/* Chat completion ID generator */

static void generate_id(char* buf, size_t size) {
    static int counter = 0;
    snprintf(buf, size, "chatcmpl-%ld-%d", (long)time(NULL), counter++);
}

/* Route handlers */

static void handle_health(int fd) {
    send_http_response(fd, 200, "OK", "application/json", "{\"status\":\"ok\"}");
}

static void handle_models(int fd, CMLOpenAIServer* srv) {
    char body[1024];
    snprintf(body, sizeof(body),
             "{\"object\":\"list\",\"data\":[{\"id\":\"%s\","
             "\"object\":\"model\",\"owned_by\":\"cml\"}]}",
             srv->model_name[0] ? srv->model_name : "cml-default");
    send_http_response(fd, 200, "OK", "application/json", body);
}

static void handle_chat_completions(int fd, CMLOpenAIServer* srv, const char* body_json) {
    CMLLLaMAModel* model = (CMLLLaMAModel*)srv->model;
    CMLTokenizer* tokenizer = (CMLTokenizer*)srv->tokenizer;

    ChatMessage msgs[MAX_MESSAGES];
    int num_msgs = parse_messages(body_json, msgs, MAX_MESSAGES);

    const char* model_p = json_find_key(body_json, "model");
    (void)model_p;

    float temperature = json_read_float(json_find_key(body_json, "temperature"), srv->temperature);
    float top_p = json_read_float(json_find_key(body_json, "top_p"), srv->top_p);
    int max_tokens = json_read_int(json_find_key(body_json, "max_tokens"), srv->max_tokens);
    bool stream = json_read_bool(json_find_key(body_json, "stream"), false);

    if (!model || !tokenizer || num_msgs == 0) {
        const char* err = "{\"error\":{\"message\":\"Model not loaded or empty messages\","
                          "\"type\":\"invalid_request_error\"}}";
        send_http_response(fd, 400, "Bad Request", "application/json", err);
        return;
    }

    char prompt[32768] = {0};
    size_t prompt_off = 0;
    for (int i = 0; i < num_msgs; i++) {
        int written = snprintf(prompt + prompt_off, sizeof(prompt) - prompt_off,
                               "[%s]: %s\n", msgs[i].role, msgs[i].content);
        if (written > 0) prompt_off += (size_t)written;
    }

    CMLGenerationConfig gen_config = cml_generation_default_config();
    gen_config.temperature = temperature;
    gen_config.top_p = top_p;
    gen_config.max_new_tokens = max_tokens;
    gen_config.do_sample = (temperature > 0.0f);

    char comp_id[64];
    generate_id(comp_id, sizeof(comp_id));
    long created = (long)time(NULL);

    if (stream) {
        char hdr[512];
        int hdr_len = snprintf(hdr, sizeof(hdr),
                               "HTTP/1.1 200 OK\r\n"
                               "Content-Type: text/event-stream\r\n"
                               "Cache-Control: no-cache\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Connection: keep-alive\r\n"
                               "\r\n");
        send(fd, hdr, (size_t)hdr_len, MSG_NOSIGNAL);

        int num_prompt_tokens = 0;
        int* prompt_tokens = cml_tokenizer_encode(tokenizer, prompt, &num_prompt_tokens);
        if (!prompt_tokens || num_prompt_tokens == 0) {
            send_sse_chunk(fd, "[DONE]");
            free(prompt_tokens);
            return;
        }

        cml_llama_reset(model);
        int total_generated = 0;

        for (int i = 0; i < max_tokens; i++) {
            int seq_len = (i == 0) ? num_prompt_tokens : 1;
            const int* input_tokens = (i == 0) ? prompt_tokens : &prompt_tokens[num_prompt_tokens + i - 1];

            Tensor* logits = cml_llama_forward(model, input_tokens, seq_len);
            if (!logits) break;

            int token_id = cml_llama_sample_token(logits, &gen_config);
            tensor_free(logits);

            if (token_id == gen_config.eos_token_id) break;

            char* token_text = cml_tokenizer_decode(tokenizer, &token_id, 1);
            if (!token_text) break;

            char chunk_json[16384];
            char escaped[8192];
            size_t ei = 0;
            for (size_t ti = 0; token_text[ti] && ei < sizeof(escaped) - 2; ti++) {
                if (token_text[ti] == '"' || token_text[ti] == '\\') {
                    escaped[ei++] = '\\';
                }
                if (token_text[ti] == '\n') {
                    escaped[ei++] = '\\'; escaped[ei++] = 'n'; continue;
                }
                escaped[ei++] = token_text[ti];
            }
            escaped[ei] = '\0';

            snprintf(chunk_json, sizeof(chunk_json),
                     "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                     "\"created\":%ld,\"model\":\"%s\","
                     "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"%s\"},"
                     "\"finish_reason\":null}]}",
                     comp_id, created,
                     srv->model_name[0] ? srv->model_name : "cml-default",
                     escaped);

            send_sse_chunk(fd, chunk_json);
            free(token_text);
            total_generated++;
        }

        char done_json[512];
        snprintf(done_json, sizeof(done_json),
                 "{\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
                 "\"created\":%ld,\"model\":\"%s\","
                 "\"choices\":[{\"index\":0,\"delta\":{},"
                 "\"finish_reason\":\"stop\"}]}",
                 comp_id, created,
                 srv->model_name[0] ? srv->model_name : "cml-default");
        send_sse_chunk(fd, done_json);
        send_sse_chunk(fd, "[DONE]");

        free(prompt_tokens);
    } else {
        CMLGenerationResult* result = cml_llama_generate(model, prompt, &gen_config);
        if (!result || !result->text) {
            const char* err = "{\"error\":{\"message\":\"Generation failed\","
                              "\"type\":\"server_error\"}}";
            send_http_response(fd, 500, "Internal Server Error", "application/json", err);
            if (result) cml_generation_result_free(result);
            return;
        }

        char escaped[32768];
        size_t ei = 0;
        for (size_t i = 0; result->text[i] && ei < sizeof(escaped) - 2; i++) {
            if (result->text[i] == '"' || result->text[i] == '\\') {
                escaped[ei++] = '\\';
            }
            if (result->text[i] == '\n') {
                escaped[ei++] = '\\'; escaped[ei++] = 'n'; continue;
            }
            escaped[ei++] = result->text[i];
        }
        escaped[ei] = '\0';

        int prompt_tokens_count = 0;
        int* pt = cml_tokenizer_encode(tokenizer, prompt, &prompt_tokens_count);
        free(pt);

        char resp[65536];
        snprintf(resp, sizeof(resp),
                 "{\"id\":\"%s\",\"object\":\"chat.completion\","
                 "\"created\":%ld,\"model\":\"%s\","
                 "\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                 "\"content\":\"%s\"},\"finish_reason\":\"stop\"}],"
                 "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,"
                 "\"total_tokens\":%d}}",
                 comp_id, created,
                 srv->model_name[0] ? srv->model_name : "cml-default",
                 escaped,
                 prompt_tokens_count, result->num_tokens,
                 prompt_tokens_count + result->num_tokens);

        send_http_response(fd, 200, "OK", "application/json", resp);
        cml_generation_result_free(result);
    }
}

/* Request dispatch */

static void handle_request(int fd, CMLOpenAIServer* srv, const char* request) {
    const char* body = strstr(request, "\r\n\r\n");
    if (body) body += 4;

    if (strncmp(request, "GET /health", 11) == 0) {
        handle_health(fd);
    } else if (strncmp(request, "GET /v1/models", 14) == 0) {
        handle_models(fd, srv);
    } else if (strncmp(request, "POST /v1/chat/completions", 25) == 0) {
        if (!body) {
            send_http_response(fd, 400, "Bad Request", "application/json",
                               "{\"error\":{\"message\":\"Missing body\"}}");
            return;
        }
        handle_chat_completions(fd, srv, body);
    } else if (strncmp(request, "OPTIONS ", 8) == 0) {
        char cors[] = "";
        char hdr[256];
        int hdr_len = snprintf(hdr, sizeof(hdr),
                               "HTTP/1.1 204 No Content\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                               "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                               "Connection: close\r\n\r\n");
        send(fd, hdr, (size_t)hdr_len, MSG_NOSIGNAL);
        (void)cors;
    } else {
        send_http_response(fd, 404, "Not Found", "application/json",
                           "{\"error\":{\"message\":\"Not found\"}}");
    }
}

/* Server lifecycle */

CMLOpenAIServer* cml_openai_server_create(int port) {
    if (port <= 0) return NULL;

    CMLOpenAIServer* srv = (CMLOpenAIServer*)calloc(1, sizeof(CMLOpenAIServer));
    if (!srv) return NULL;

    srv->port = port;
    srv->listen_fd = -1;
    srv->max_tokens = 256;
    srv->temperature = 0.8f;
    srv->top_p = 0.9f;

    return srv;
}

int cml_openai_server_load_model(CMLOpenAIServer* srv, const char* model_path) {
    if (!srv || !model_path) return -1;

    strncpy(srv->model_path, model_path, sizeof(srv->model_path) - 1);

    const char* basename = strrchr(model_path, '/');
    basename = basename ? basename + 1 : model_path;
    strncpy(srv->model_name, basename, sizeof(srv->model_name) - 1);

    CMLLLaMAConfig config = cml_llama_config_7b();
    CMLLLaMAModel* model = cml_llama_create(&config);
    if (!model) {
        LOG_ERROR("openai_api: failed to create model");
        return -1;
    }

    if (cml_llama_load_gguf(model, model_path) != 0) {
        LOG_ERROR("openai_api: failed to load weights from %s", model_path);
        cml_llama_free(model);
        return -1;
    }

    srv->model = model;
    srv->tokenizer = model->tokenizer;
    LOG_INFO("OpenAI API: loaded model from %s", model_path);
    return 0;
}

int cml_openai_server_run(CMLOpenAIServer* srv) {
    if (!srv) return -1;

    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->listen_fd < 0) {
        LOG_ERROR("openai_api: socket() failed: %s", strerror(errno));
        return -1;
    }

    int opt = 1;
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)srv->port);

    if (bind(srv->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        LOG_ERROR("openai_api: bind() port %d failed: %s", srv->port, strerror(errno));
        close(srv->listen_fd);
        srv->listen_fd = -1;
        return -1;
    }

    if (listen(srv->listen_fd, 16) != 0) {
        LOG_ERROR("openai_api: listen() failed: %s", strerror(errno));
        close(srv->listen_fd);
        srv->listen_fd = -1;
        return -1;
    }

    srv->running = true;
    LOG_INFO("OpenAI-compatible API server listening on port %d", srv->port);
    LOG_INFO("  POST /v1/chat/completions");
    LOG_INFO("  GET  /v1/models");
    LOG_INFO("  GET  /health");

    char* buf = (char*)malloc(HTTP_BUF_SIZE);
    if (!buf) {
        LOG_ERROR("openai_api: buffer allocation failed");
        return -1;
    }

    while (srv->running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(srv->listen_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) {
            if (!srv->running) break;
            if (errno == EINTR) continue;
            LOG_ERROR("openai_api: accept() failed: %s", strerror(errno));
            continue;
        }

        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        int n = recv_http_request(client_fd, buf, HTTP_BUF_SIZE);
        if (n > 0) {
            handle_request(client_fd, srv, buf);
        }

        close(client_fd);
    }

    free(buf);
    return 0;
}

void cml_openai_server_stop(CMLOpenAIServer* srv) {
    if (!srv) return;
    srv->running = false;
    if (srv->listen_fd >= 0) {
        shutdown(srv->listen_fd, SHUT_RDWR);
    }
}

void cml_openai_server_free(CMLOpenAIServer* srv) {
    if (!srv) return;
    cml_openai_server_stop(srv);
    if (srv->listen_fd >= 0) {
        close(srv->listen_fd);
    }
    if (srv->model) {
        cml_llama_free((CMLLLaMAModel*)srv->model);
    }
    free(srv);
}
