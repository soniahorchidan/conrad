# 3p precision: best precision at recall >= 0.60 (20% incompleteness)

| Dataset | neo4j | neural (best θ) | hybrid (best θ) | conrad (best α) |
|---|---|---|---|---|
| FB15k-237 | Fails (0.45) | Fails (0.54) | 0.38 (θ=0.3) | **0.89 (α=0.8)** |
| NELL-995 | Fails (0.30) | Fails (0.49) | 0.65 (θ=0.3) | **0.85 (α=0.8)** |
