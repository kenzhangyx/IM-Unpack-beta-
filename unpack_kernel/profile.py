from kernel import construct_matrix, profile, row_unpack, col_unpack, both_unpack
import torch

results = []
for L in [4096, 8192, 16384, 32768]:
    for N in [4096, 8192, 16384, 32768]:
        for D in [4096, 8192, 16384, 32768]:
            res = {"n": L, "d": D, "h": N}

            A = construct_matrix(D, L, 8, 1.1)
            B = construct_matrix(D, N, 8, 1.1)
            
            clone_t = profile(lambda: A.clone()) + profile(lambda: B.clone())
            matmul1_t = profile(lambda: torch.matmul(A.T, B))
            
            A, B = A.half(), B.half()
            matmul2_t = profile(lambda: torch.matmul(A.T, B))

            res["fp32_clone"] = round(clone_t * 1000, 2)
            res["fp32_matmul"] = round(matmul1_t * 1000, 2)
            res["fp16_matmul"] = round(matmul2_t * 1000, 2)
            
            for bit in [2, 4, 8]:
                for ratio in [1.1, 1.2, 1.4]:
        
                    A = construct_matrix(L, D, bit, ratio)
                    B = construct_matrix(N, D, bit, ratio)

                    row_unpack_t = profile(lambda: row_unpack(A, scale = 1, bits = bit)) + profile(lambda: row_unpack(B, scale = 1, bits = bit))
                    
                    A = construct_matrix(D, L, bit, ratio)
                    B = construct_matrix(D, N, bit, ratio)

                    col_unpack_t = profile(lambda: col_unpack(A, B, scale = 1, bits = bit))

                    A = construct_matrix(L, D, bit, ratio)
                    B = construct_matrix(N, D, bit, ratio)

                    both_unpack_t = profile(lambda: both_unpack(A, B, scale=1, bits=bit))

                    res[f"fp32_row_unpack(bits={bit},r={ratio ** 2:.2f})"] = round(row_unpack_t * 1000, 2)
                    res[f"fp32_col_unpack(bits={bit},r={ratio ** 2:.2f})"] = round(col_unpack_t * 1000, 2)
                    res[f"fp32_both_unpack(bits={bit},r={ratio ** 2:.2f})"] = round(both_unpack_t * 1000, 2)
            results.append(res)
            print(results[-1])

df = pd.DataFrame(results)
df.to_csv("profiles.csv")