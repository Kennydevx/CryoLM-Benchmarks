# FASE 1: CONFIRMAÇÃO CIENTÍFICA PRÉ-EXECUÇÃO
## Verificação Estrutural de Tensores e Paridade Exata do Paper Publicado

### 1. Resumo Executivo da Auditoria de Paridade
| Métrica Auditada | Dense Transformer Baseline | CryoLM GSA (Modo 11) | Status |
| :--- | :---: | :---: | :--- |
| **Configuração** | 4L, d_model=336, 8 heads, SDPA | 4L, d_state=256, d_hidden=688, GSA | Conforme Paper |
| **Total de Parâmetros** | **6,769,056** | **7,088,385** | Δ = +319,329 (4.72%) |
| **Parâmetros Treináveis** | **6,769,056** | **7,088,385** | 100% Treináveis |
| **Embeddings (TokenEmb)** | 672,000 | 512,000 | CryoLM -23.81% |
| **Output Head (LMHead)** | 672,000 | 512,000 | CryoLM -23.81% |
| **Final Normalization** | 672 | 256 | CryoLM -61.90% |
| **Camada 0 (Attn/Rec + MLP)** | 1,356,096 | 1,516,032 | CryoLM +11.79% |
| **Camada 1 (Attn/Rec + MLP)** | 1,356,096 | 1,516,032 | CryoLM +11.79% |
| **Camada 2 (Attn/Rec + MLP)** | 1,356,096 | 1,516,032 | CryoLM +11.79% |
| **Camada 3 (Attn/Rec + MLP)** | 1,356,096 | 1,516,032 | CryoLM +11.79% |
| **Bytes por Parâmetro** | 4 bytes (Float32) | 4 bytes (Float32) | Padrão IEEE 754 |
| **Memória Total de Pesos** | 25.82 MB | 27.04 MB | Δ = +1.22 MB |

### 2. Teste Obrigatório de Comparabilidade de Parâmetros
- **Regra Estrita de Paridade Científica:** $|\Delta| / \text{Dense} \le 5.0\%$
- **Diferença Real:** $|7,088,385 - 6,769,056| = 319,329$
- **Variação Calculada:** **4.72%**
- **Razão de Parâmetros:** **1.0472x**
- **Veredito:** **APROVADO (A alegação de contagem de parâmetros comparável é matematicamente válida)**

### 3. Tabela Completa de Tensores — CryoLM (4L, d_state=256, d_hidden=688)
| Nome do Tensor | Formato (Shape) | Elementos (Numel) | Dtype | Bytes/Elem | Memória Total (Bytes) | Treinável |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `graph_alpha` | `[1]` | 1 | Float32 (Single) | 4 | 4 | Sim |
| `token_emb.embedding.weight` | `[2000, 256]` | 512,000 | Float32 (Single) | 4 | 2,048,000 | Sim |
| `layers.0.log_decay` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.0.norm.weight` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.0.W_q.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.0.W_k.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.0.W_v.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.0.W_o.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.0.mlp.w1.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.0.mlp.w2.weight` | `[512, 688]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.0.mlp.w3.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.0.down_proj.weight` | `[256, 512]` | 131,072 | Float32 (Single) | 4 | 524,288 | Sim |
| `layers.0.mem_proj.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.1.log_decay` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.1.norm.weight` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.1.W_q.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.1.W_k.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.1.W_v.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.1.W_o.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.1.mlp.w1.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.1.mlp.w2.weight` | `[512, 688]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.1.mlp.w3.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.1.down_proj.weight` | `[256, 512]` | 131,072 | Float32 (Single) | 4 | 524,288 | Sim |
| `layers.1.mem_proj.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.2.log_decay` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.2.norm.weight` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.2.W_q.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.2.W_k.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.2.W_v.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.2.W_o.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.2.mlp.w1.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.2.mlp.w2.weight` | `[512, 688]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.2.mlp.w3.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.2.down_proj.weight` | `[256, 512]` | 131,072 | Float32 (Single) | 4 | 524,288 | Sim |
| `layers.2.mem_proj.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.3.log_decay` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.3.norm.weight` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `layers.3.W_q.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.3.W_k.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.3.W_v.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.3.W_o.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `layers.3.mlp.w1.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.3.mlp.w2.weight` | `[512, 688]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.3.mlp.w3.weight` | `[688, 512]` | 352,256 | Float32 (Single) | 4 | 1,409,024 | Sim |
| `layers.3.down_proj.weight` | `[256, 512]` | 131,072 | Float32 (Single) | 4 | 524,288 | Sim |
| `layers.3.mem_proj.weight` | `[256, 256]` | 65,536 | Float32 (Single) | 4 | 262,144 | Sim |
| `final_norm.weight` | `[256]` | 256 | Float32 (Single) | 4 | 1,024 | Sim |
| `lm_head.projection.weight` | `[2000, 256]` | 512,000 | Float32 (Single) | 4 | 2,048,000 | Sim |

### 4. Tabela Completa de Tensores — Dense Transformer Baseline (4L, d_model=336, 8 heads)
| Nome do Tensor | Formato (Shape) | Elementos (Numel) | Dtype | Bytes/Elem | Memória Total (Bytes) | Treinável |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `token_emb.embedding.weight` | `[2000, 336]` | 672,000 | Float32 (Single) | 4 | 2,688,000 | Sim |
| `lm_head.projection.weight` | `[2000, 336]` | 672,000 | Float32 (Single) | 4 | 2,688,000 | Sim |
| `final_norm.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `final_norm.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.0.attn.q_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.0.attn.k_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.0.attn.v_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.0.attn.out_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.0.mlp.w1.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.0.mlp.w2.weight` | `[336, 896]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.0.mlp.w3.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.0.norm1.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.0.norm1.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.0.norm2.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.0.norm2.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.1.attn.q_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.1.attn.k_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.1.attn.v_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.1.attn.out_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.1.mlp.w1.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.1.mlp.w2.weight` | `[336, 896]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.1.mlp.w3.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.1.norm1.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.1.norm1.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.1.norm2.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.1.norm2.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.2.attn.q_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.2.attn.k_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.2.attn.v_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.2.attn.out_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.2.mlp.w1.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.2.mlp.w2.weight` | `[336, 896]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.2.mlp.w3.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.2.norm1.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.2.norm1.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.2.norm2.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.2.norm2.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.3.attn.q_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.3.attn.k_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.3.attn.v_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.3.attn.out_proj.weight` | `[336, 336]` | 112,896 | Float32 (Single) | 4 | 451,584 | Sim |
| `layers.3.mlp.w1.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.3.mlp.w2.weight` | `[336, 896]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.3.mlp.w3.weight` | `[896, 336]` | 301,056 | Float32 (Single) | 4 | 1,204,224 | Sim |
| `layers.3.norm1.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.3.norm1.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.3.norm2.weight` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |
| `layers.3.norm2.bias` | `[336]` | 336 | Float32 (Single) | 4 | 1,344 | Sim |