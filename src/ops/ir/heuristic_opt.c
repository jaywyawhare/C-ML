#include "ops/ir/heuristic_opt.h"
#include "ops/ir/linearize.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

static CMLHeuristicConfig g_heuristic_config = {
    .max_local_size = CML_HEURISTIC_DEFAULT_LOCAL_SIZE,
    .preferred_vec_width = CML_HEURISTIC_DEFAULT_VEC_WIDTH,
    .use_local_memory = true,
};

void cml_heuristic_set_config(CMLHeuristicConfig* config) {
    if (!config) return;
    g_heuristic_config = *config;
}

CMLHeuristicConfig cml_heuristic_get_config(void) {
    return g_heuristic_config;
}

typedef enum {
    KERNEL_ELEMENTWISE,
    KERNEL_REDUCE,
    KERNEL_MATMUL,
    KERNEL_CONV,
    KERNEL_UNKNOWN,
} KernelKind;

static KernelKind classify_program(const struct LinearProgram* prog) {
    bool has_reduce = false;
    bool has_matmul = false;
    bool has_conv = false;

    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind != LINOP_COMPUTE) continue;
        UOpType uop = prog->ops[i].uop;
        switch (uop) {
        case UOP_SUM: case UOP_MAX_REDUCE: case UOP_MEAN:
        case UOP_PROD: case UOP_MIN_REDUCE: case UOP_VAR:
        case UOP_STD: case UOP_ANY: case UOP_ALL:
        case UOP_LOGSUMEXP: case UOP_ARGMAX: case UOP_ARGMIN:
            has_reduce = true;
            break;
        case UOP_MATMUL:
            has_matmul = true;
            break;
        case UOP_CONV2D:
            has_conv = true;
            break;
        default:
            break;
        }
    }

    if (has_matmul) return KERNEL_MATMUL;
    if (has_conv) return KERNEL_CONV;
    if (has_reduce) return KERNEL_REDUCE;
    return KERNEL_ELEMENTWISE;
}

static int find_reduce_axis(const struct LinearProgram* prog) {
    for (int i = 0; i < prog->num_ops; i++) {
        if (prog->ops[i].kind == LINOP_LOOP && prog->ops[i].loop_extent > 1)
            return prog->ops[i].loop_axis;
    }
    return prog->num_axes > 0 ? prog->num_axes - 1 : 0;
}

static int innermost_axis(const struct LinearProgram* prog) {
    return prog->num_axes > 0 ? prog->num_axes - 1 : 0;
}

static int pick_tile(int extent, int preferred) {
    if (extent <= 0) return 1;
    int tile = preferred;
    while (tile > 1 && extent % tile != 0)
        tile /= 2;
    return tile;
}

static int pick_power2_tile(int extent, int max_tile) {
    int tile = 1;
    while (tile * 2 <= max_tile && tile * 2 <= extent && extent % (tile * 2) == 0)
        tile *= 2;
    return tile;
}

static CMLOptList* optimize_elementwise(const struct LinearProgram* prog,
                                        const CMLHeuristicConfig* cfg) {
    CMLOptList* opts = cml_opt_list_create();
    if (!opts) return NULL;

    int inner = innermost_axis(prog);
    int extent = (inner < prog->num_axes) ? prog->loop_axes[inner] : 0;

    if (extent > 1) {
        int vec = pick_tile(extent, cfg->preferred_vec_width);
        if (vec > 1)
            cml_opt_list_add(opts, OPT_UPCAST, inner, vec);
    }

    int best_group_axis = -1;
    int best_group_extent = 0;
    for (int ax = 0; ax < prog->num_axes; ax++) {
        if (prog->loop_axes[ax] > best_group_extent) {
            best_group_extent = prog->loop_axes[ax];
            best_group_axis = ax;
        }
    }

    if (best_group_axis >= 0 && best_group_extent > 1) {
        int vec = (inner < prog->num_axes) ? pick_tile(prog->loop_axes[inner], cfg->preferred_vec_width) : 1;
        if (vec < 1) vec = 1;
        int group_target = cfg->max_local_size / (vec > 0 ? vec : 1);
        int group_size = pick_power2_tile(best_group_extent, group_target);
        if (group_size > 1)
            cml_opt_list_add(opts, OPT_GROUP, best_group_axis, group_size);
    }

    return opts;
}

static CMLOptList* optimize_reduce(const struct LinearProgram* prog,
                                   const CMLHeuristicConfig* cfg) {
    CMLOptList* opts = cml_opt_list_create();
    if (!opts) return NULL;

    int reduce_ax = find_reduce_axis(prog);
    int reduce_extent = (reduce_ax < prog->num_axes) ? prog->loop_axes[reduce_ax] : 0;

    if (cfg->use_local_memory && reduce_extent > 1) {
        int local_tile = pick_power2_tile(reduce_extent, 64);
        if (local_tile < 32 && reduce_extent >= 32 && reduce_extent % 32 == 0)
            local_tile = 32;
        if (local_tile > 1)
            cml_opt_list_add(opts, OPT_LOCAL, reduce_ax, local_tile);
    }

    int inner = innermost_axis(prog);
    int inner_extent = (inner < prog->num_axes) ? prog->loop_axes[inner] : 0;

    if (inner != reduce_ax && inner_extent > 1) {
        int unroll = pick_tile(inner_extent, 4);
        if (unroll > 1)
            cml_opt_list_add(opts, OPT_UNROLL, inner, unroll);
    } else if (reduce_extent > 4) {
        int after_local = reduce_extent;
        for (int i = 0; i < opts->num_opts; i++) {
            if (opts->opts[i].axis == reduce_ax && opts->opts[i].type == OPT_LOCAL)
                after_local = reduce_extent / opts->opts[i].amount;
        }
        if (after_local > 1) {
            int unroll = pick_tile(after_local, 4);
            if (unroll > 1)
                cml_opt_list_add(opts, OPT_UNROLL, reduce_ax, unroll);
        }
    }

    return opts;
}

static CMLOptList* optimize_matmul(const struct LinearProgram* prog,
                                   const CMLHeuristicConfig* cfg) {
    CMLOptList* opts = cml_opt_list_create();
    if (!opts) return NULL;

    int ax_m = -1, ax_n = -1, ax_k = -1;
    if (prog->num_axes >= 3) {
        ax_m = 0;
        ax_n = 1;
        ax_k = 2;
    } else if (prog->num_axes == 2) {
        ax_m = 0;
        ax_n = 1;
    } else if (prog->num_axes == 1) {
        ax_m = 0;
    }

    if (cfg->use_local_memory) {
        if (ax_m >= 0 && prog->loop_axes[ax_m] > 1) {
            int tile = pick_power2_tile(prog->loop_axes[ax_m], 16);
            if (tile > 1)
                cml_opt_list_add(opts, OPT_LOCAL, ax_m, tile);
        }
        if (ax_n >= 0 && prog->loop_axes[ax_n] > 1) {
            int tile = pick_power2_tile(prog->loop_axes[ax_n], 16);
            if (tile > 1)
                cml_opt_list_add(opts, OPT_LOCAL, ax_n, tile);
        }
    }

    if (ax_k >= 0 && prog->loop_axes[ax_k] > 1) {
        int unroll = pick_tile(prog->loop_axes[ax_k], 4);
        if (unroll > 1)
            cml_opt_list_add(opts, OPT_UNROLL, ax_k, unroll);
    }

    return opts;
}

static CMLOptList* optimize_conv(const struct LinearProgram* prog,
                                 const CMLHeuristicConfig* cfg) {
    CMLOptList* opts = cml_opt_list_create();
    if (!opts) return NULL;

    for (int ax = 0; ax < prog->num_axes && ax < 2; ax++) {
        int extent = prog->loop_axes[ax];
        if (extent <= 1) continue;
        if (cfg->use_local_memory) {
            int tile = pick_power2_tile(extent, 16);
            if (tile > 1)
                cml_opt_list_add(opts, OPT_LOCAL, ax, tile);
        }
    }

    if (prog->num_axes > 2) {
        int ch_axis = 2;
        int ch_extent = prog->loop_axes[ch_axis];
        if (ch_extent > 1) {
            int unroll = pick_tile(ch_extent, 4);
            if (unroll > 1)
                cml_opt_list_add(opts, OPT_UNROLL, ch_axis, unroll);
        }
    }

    return opts;
}

CMLOptList* cml_heuristic_optimize(struct LinearProgram* prog) {
    if (!prog || prog->num_axes == 0)
        return cml_opt_list_create();

    KernelKind kind = classify_program(prog);
    CMLOptList* opts = NULL;

    switch (kind) {
    case KERNEL_ELEMENTWISE:
        opts = optimize_elementwise(prog, &g_heuristic_config);
        break;
    case KERNEL_REDUCE:
        opts = optimize_reduce(prog, &g_heuristic_config);
        break;
    case KERNEL_MATMUL:
        opts = optimize_matmul(prog, &g_heuristic_config);
        break;
    case KERNEL_CONV:
        opts = optimize_conv(prog, &g_heuristic_config);
        break;
    default:
        opts = cml_opt_list_create();
        break;
    }

    if (opts) {
        LOG_DEBUG("Heuristic optimizer: kind=%d, generated %d opts", kind, opts->num_opts);
    }

    return opts;
}
