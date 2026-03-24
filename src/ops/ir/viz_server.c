#include "ops/ir/viz_server.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <winsock2.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET sock_t;
#  define SOCK_INVALID INVALID_SOCKET
#  define sock_close   closesocket
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#  include <unistd.h>
#  include <fcntl.h>
   typedef int sock_t;
#  define SOCK_INVALID (-1)
#  define sock_close   close
#endif

/* =========================================================================
 * Embedded HTML/JS graph viewer
 * ========================================================================= */

/* Two-part template — JSON is inserted between them. */
static const char VIZ_HTML_PRE[] =
"<!DOCTYPE html>\n"
"<html lang='en'><head><meta charset='UTF-8'/>\n"
"<title>C-ML IR Visualizer</title>\n"
"<style>\n"
"body{margin:0;font-family:monospace;background:#1e1e2e;color:#cdd6f4;}\n"
"#hdr{padding:10px 18px;background:#181825;border-bottom:1px solid #313244;}\n"
"#hdr h1{margin:0;font-size:15px;color:#89b4fa;}\n"
"#stat{font-size:11px;color:#a6adc8;margin-top:3px;}\n"
"#wrap{overflow:auto;padding:16px;}\n"
"svg text{font-size:11px;fill:#cdd6f4;}\n"
".node rect{rx:5;ry:5;stroke-width:1.5;}\n"
".elemwise rect{fill:#1a3a5c;stroke:#89b4fa;}\n"
".reduce   rect{fill:#3a2852;stroke:#cba6f7;}\n"
".movement rect{fill:#1a3a2a;stroke:#a6e3a1;}\n"
".matmul   rect{fill:#3a3018;stroke:#f9e2af;}\n"
".other    rect{fill:#28283c;stroke:#6c7086;}\n"
".edge{stroke:#585b70;stroke-width:1;fill:none;marker-end:url(#arr);}\n"
"#sb{position:fixed;right:0;top:0;width:280px;height:100%;"
"    background:#181825;border-left:1px solid #313244;"
"    padding:14px;overflow-y:auto;font-size:12px;}\n"
"#sb h2{color:#89b4fa;margin-top:0;font-size:13px;}\n"
".p{margin:3px 0;}.p span{color:#a6e3a1;}\n"
"</style></head><body>\n"
"<div id='hdr'><h1>C-ML IR Graph Visualizer</h1>"
"<div id='stat'></div></div>\n"
"<div style='display:flex'>\n"
"<div id='wrap' style='flex:1'><svg id='g'></svg></div>\n"
"<div id='sb'><h2>Node Info</h2><div id='inf'>Click a node</div></div>\n"
"</div>\n"
"<script>\nconst D=";

static const char VIZ_HTML_POST[] =
";\n"
"const svg=document.getElementById('g');\n"
"const inf=document.getElementById('inf');\n"
"document.getElementById('stat').textContent="
"  D.nodes.length+' nodes, '+D.edges.length+' edges';\n"
"// layered layout\n"
"const rk=new Map();\n"
"D.edges.forEach(e=>rk.set(e.dst,Math.max(rk.get(e.dst)||0,(rk.get(e.src)||0)+1)));\n"
"D.nodes.forEach(n=>{if(!rk.has(n.id))rk.set(n.id,0);});\n"
"const ly={};\n"
"rk.forEach((r,id)=>{if(!ly[r])ly[r]=[];ly[r].push(id);});\n"
"const NW=155,NH=42,PX=38,PY=58;\n"
"const pos={};\n"
"Object.keys(ly).forEach(r=>ly[r].forEach((id,i)=>pos[id]={x:i*(NW+PX)+16,y:r*(NH+PY)+16}));\n"
"const mxW=Math.max(...Object.values(pos).map(p=>p.x))+NW+32;\n"
"const mxH=Math.max(...Object.values(pos).map(p=>p.y))+NH+32;\n"
"svg.setAttribute('viewBox',`0 0 ${mxW} ${mxH}`);\n"
"svg.setAttribute('height',mxH);\n"
"svg.innerHTML='<defs><marker id=\"arr\" markerWidth=\"7\" markerHeight=\"7\""\
" refX=\"5\" refY=\"3\" orient=\"auto\">"\
"<path d=\"M0,0 L0,6 L7,3 z\" fill=\"#585b70\"/></marker></defs>';\n"
"D.edges.forEach(e=>{\n"
"  const s=pos[e.src],d=pos[e.dst];if(!s||!d)return;\n"
"  const p=document.createElementNS('http://www.w3.org/2000/svg','path');\n"
"  const sx=s.x+NW/2,sy=s.y+NH,dx=d.x+NW/2,dy=d.y,my=(sy+dy)/2;\n"
"  p.setAttribute('d',`M${sx},${sy} C${sx},${my} ${dx},${my} ${dx},${dy}`);\n"
"  p.setAttribute('class','edge');svg.appendChild(p);\n"
"});\n"
"D.nodes.forEach(n=>{\n"
"  const p=pos[n.id];if(!p)return;\n"
"  const g=document.createElementNS('http://www.w3.org/2000/svg','g');\n"
"  g.setAttribute('class','node '+n.cls);\n"
"  g.setAttribute('transform',`translate(${p.x},${p.y})`);\n"
"  g.style.cursor='pointer';\n"
"  const r=document.createElementNS('http://www.w3.org/2000/svg','rect');\n"
"  r.setAttribute('width',NW);r.setAttribute('height',NH);\n"
"  const t1=document.createElementNS('http://www.w3.org/2000/svg','text');\n"
"  t1.setAttribute('x',NW/2);t1.setAttribute('y',17);\n"
"  t1.setAttribute('text-anchor','middle');t1.setAttribute('font-weight','bold');\n"
"  t1.textContent=n.op;\n"
"  const t2=document.createElementNS('http://www.w3.org/2000/svg','text');\n"
"  t2.setAttribute('x',NW/2);t2.setAttribute('y',32);\n"
"  t2.setAttribute('text-anchor','middle');t2.setAttribute('font-size','10');\n"
"  t2.setAttribute('fill','#a6adc8');\n"
"  t2.textContent=(n.shape?'['+n.shape.join('x')+'] ':'')+( n.dtype||'');\n"
"  g.appendChild(r);g.appendChild(t1);g.appendChild(t2);\n"
"  g.addEventListener('click',()=>{\n"
"    inf.innerHTML='<div class=\"p\">id: <span>'+n.id+'</span></div>'\n"
"      +'<div class=\"p\">op: <span>'+n.op+'</span></div>'\n"
"      +(n.shape?'<div class=\"p\">shape: <span>['+n.shape.join(', ')+']</span></div>':'')\n"
"      +(n.dtype?'<div class=\"p\">dtype: <span>'+n.dtype+'</span></div>':'');\n"
"  });\n"
"  svg.appendChild(g);\n"
"});\n"
"</script></body></html>\n";

/* =========================================================================
 * JSON serialisation
 * ========================================================================= */

static const char* dtype_name(DType d) {
    switch (d) {
        case DTYPE_FLOAT32:  return "float32";
        case DTYPE_FLOAT16:  return "float16";
        case DTYPE_BFLOAT16: return "bfloat16";
        case DTYPE_INT32:    return "int32";
        case DTYPE_INT64:    return "int64";
        case DTYPE_INT8:     return "int8";
        case DTYPE_UINT8:    return "uint8";
        case DTYPE_BOOL:     return "bool";
        default:             return "unknown";
    }
}

static const char* op_class(UOpType op) {
    switch (op) {
        case UOP_ADD: case UOP_SUB: case UOP_MUL: case UOP_DIV:
        case UOP_NEG: case UOP_EXP: case UOP_LOG: case UOP_SQRT:
        case UOP_SIGMOID: case UOP_TANH: case UOP_ABS:
            return "elemwise";
        case UOP_SUM: case UOP_MEAN: case UOP_MAX_REDUCE:
        case UOP_PROD: case UOP_ARGMAX: case UOP_ARGMIN:
            return "reduce";
        case UOP_RESHAPE: case UOP_PERMUTE: case UOP_EXPAND:
        case UOP_SLICE: case UOP_STRIDE:
            return "movement";
        case UOP_MATMUL:
            return "matmul";
        default:
            return "other";
    }
}

char* viz_graph_to_json(CMLGraph_t ir) {
    if (!ir) return NULL;

    /* Collect nodes */
    int n = ir->node_count;
    struct IRNode** nodes = malloc((size_t)(n + 1) * sizeof(struct IRNode*));
    if (!nodes) return NULL;
    int cnt = 0;
    for (struct IRNode* cur = ir->head; cur && cnt < n; cur = cur->next)
        nodes[cnt++] = cur;
    n = cnt;

    size_t cap = (size_t)(n * 256) + 2048;
    char* buf = malloc(cap);
    if (!buf) { free(nodes); return NULL; }
    size_t pos = 0;

#define A(...) do { int _r = snprintf(buf+pos, cap-pos, __VA_ARGS__); \
                    if (_r > 0) pos += (size_t)_r; } while(0)

    A("{\"nodes\":[");
    for (int i = 0; i < n; ++i) {
        struct IRNode* node = nodes[i];
        if (i > 0) A(",");
        A("{\"id\":%d,\"op\":\"%s\",\"cls\":\"%s\"",
          i, uop_type_to_string(node->type), op_class(node->type));
        if (node->output) {
            A(",\"dtype\":\"%s\"", dtype_name(node->output->dtype));
            if (node->output->ndim > 0) {
                A(",\"shape\":[");
                for (int d = 0; d < node->output->ndim; ++d) {
                    if (d > 0) A(",");
                    A("%d", node->output->shape[d]);
                }
                A("]");
            }
        }
        A("}");
    }
    A("],\"edges\":[");

    bool first_edge = true;
    for (int i = 0; i < n; ++i) {
        struct IRNode* node = nodes[i];
        if (!node->inputs) continue;
        for (int s = 0; s < node->num_inputs; ++s) {
            Tensor* inp = node->inputs[s];
            if (!inp || !inp->ir_node) continue;
            for (int j = 0; j < n; ++j) {
                if (nodes[j] == (struct IRNode*)inp->ir_node) {
                    if (!first_edge) A(",");
                    A("{\"src\":%d,\"dst\":%d,\"slot\":%d}", j, i, s);
                    first_edge = false;
                    break;
                }
            }
        }
    }
    A("]}");
#undef A
    free(nodes);
    return buf;
}

int viz_export_html(CMLGraph_t ir, const char* path) {
    if (!path) return -1;
    char* json = viz_graph_to_json(ir);
    if (!json) return -1;
    FILE* f = fopen(path, "w");
    if (!f) { free(json); return -1; }
    fputs(VIZ_HTML_PRE, f);
    fputs(json, f);
    fputs(VIZ_HTML_POST, f);
    fclose(f);
    free(json);
    LOG_INFO("Exported IR graph to %s", path);
    return 0;
}

/* =========================================================================
 * HTTP server
 * ========================================================================= */

struct VizServer {
    sock_t     listen_fd;
    int        port;
    CMLGraph_t ir;
    char*      json_cache;
};

static void set_nonblocking(sock_t fd) {
#ifdef _WIN32
    u_long m = 1; ioctlsocket(fd, FIONBIO, &m);
#else
    int fl = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, fl | O_NONBLOCK);
#endif
}

VizServer* viz_server_start(CMLGraph_t ir, int port) {
#ifdef _WIN32
    WSADATA wd; WSAStartup(MAKEWORD(2,2), &wd);
#endif
    sock_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCK_INVALID) return NULL;
    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons((uint16_t)port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0 ||
        listen(fd, 8) != 0) {
        sock_close(fd); return NULL;
    }
    set_nonblocking(fd);

    if (port == 0) {
        struct sockaddr_in b; socklen_t l = sizeof(b);
        getsockname(fd, (struct sockaddr*)&b, &l);
        port = ntohs(b.sin_port);
    }

    VizServer* srv = calloc(1, sizeof(VizServer));
    if (!srv) { sock_close(fd); return NULL; }
    srv->listen_fd  = fd;
    srv->port       = port;
    srv->ir         = ir;
    srv->json_cache = ir ? viz_graph_to_json(ir) : NULL;
    LOG_INFO("IR viz server: http://localhost:%d", port);
    return srv;
}

void viz_server_update(VizServer* srv, CMLGraph_t ir) {
    if (!srv) return;
    srv->ir = ir;
    free(srv->json_cache);
    srv->json_cache = ir ? viz_graph_to_json(ir) : NULL;
}

static void handle_request(VizServer* srv, sock_t cfd) {
    char req[1024];
    recv(cfd, req, sizeof(req)-1, 0);
    const char* json = srv->json_cache ? srv->json_cache : "{}";
    size_t pre_len  = strlen(VIZ_HTML_PRE);
    size_t json_len = strlen(json);
    size_t post_len = strlen(VIZ_HTML_POST);
    size_t body_len = pre_len + json_len + post_len;
    char hdr[256];
    int hlen = snprintf(hdr, sizeof(hdr),
                        "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\n"
                        "Content-Length: %zu\r\nConnection: close\r\n\r\n",
                        body_len);
    send(cfd, hdr, hlen, 0);
    send(cfd, VIZ_HTML_PRE,  (int)pre_len,  0);
    send(cfd, json,          (int)json_len, 0);
    send(cfd, VIZ_HTML_POST, (int)post_len, 0);
    sock_close(cfd);
}

int viz_server_poll(VizServer* srv) {
    if (!srv) return 0;
    int handled = 0;
    struct sockaddr_in ca; socklen_t cl = sizeof(ca);
    sock_t cfd;
    while ((cfd = accept(srv->listen_fd, (struct sockaddr*)&ca, &cl)) != SOCK_INVALID) {
        handle_request(srv, cfd);
        ++handled;
    }
    return handled;
}

void viz_server_stop(VizServer* srv) {
    if (!srv) return;
    sock_close(srv->listen_fd);
    free(srv->json_cache);
    free(srv);
#ifdef _WIN32
    WSACleanup();
#endif
}

int viz_server_port(const VizServer* srv) {
    return srv ? srv->port : -1;
}
