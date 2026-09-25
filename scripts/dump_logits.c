/* Reference-logits harness for llama2.c checkpoints (TinyStories).
 *
 * run.c is compiled with -DTESTING so it provides everything except `main`;
 * this file supplies struct declarations (matching run.c) and a `main` that
 * feeds token ids straight into `forward`, skipping the tokenizer entirely.
 *
 * Build (from the repository root, after fetching run.c into target/):
 *   clang -O2 -DTESTING -c target/run.c -o target/run.o
 *   clang -O2 scripts/dump_logits.c target/run.o -lm -o target/dump_logits
 *
 * Run:
 *   target/dump_logits target/stories15M.bin 1,299,1234,5678 > target/ref_logits.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
} Config;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* rms_ffn_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
} TransformerWeights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits;
    float* key_cache;
    float* value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    int fd;
    float* data;
    ssize_t file_size;
} Transformer;

void build_transformer(Transformer* t, char* checkpoint_path);
float* forward(Transformer* transformer, int token, int pos);

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.bin id,id,...\n", argv[0]);
        return 1;
    }
    Transformer t;
    build_transformer(&t, argv[1]);

    int ids[1024];
    int n = 0;
    for (char* tok = strtok(argv[2], ","); tok != NULL && n < 1024; tok = strtok(NULL, ",")) {
        ids[n++] = atoi(tok);
    }
    if (n < 1) {
        fprintf(stderr, "no token ids given\n");
        return 1;
    }

    float* logits = NULL;
    for (int pos = 0; pos < n; pos++) {
        logits = forward(&t, ids[pos], pos);
    }
    for (int i = 0; i < t.config.vocab_size; i++) {
        printf("%.9g\n", logits[i]);
    }
    return 0;
}
