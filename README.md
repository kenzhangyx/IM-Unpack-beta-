# Official Repo of IM-Unpack

``unpack_kernel`` contains a cuda implementation of UnpackRow and UnpackCol for FP32 matrices A and B to evaluate the runtime overhead of Im-Unpack algorithms. 
The supported unpack bit widths are 2, 4, 8 by packing the bit representation of 16 x INT2 or 8 x INT4 or 4 x INT8 into an INT32. For other bit-widths, an efficient implementation would need some hardware support. 

``unpack.py`` contains PyTorch implementations of Unpack algorithms and all other algorithms (such as ScaledMatmul) for supporting GEMM between unpacked matrices. ``check_unpack_correct.py`` checks the GEMM between unpacked matrices produce exactly the same output as GEMM between the original matrices. These two files are only for domestration purpose and correctness check, so we don't expect the speed will be fast. 

## Training

``training`` contains implementations for quantized training. ``training/roberta`` contains training implementations of roberta pretraining. Use 
```
python3 main.py --config cfgs/small/prenorm_qmm-q95n255.py
```
to run pretraining. 

``training/vit`` contains model implementation for quantized vit training. Place ``training/vit/vision_transformer_quantize.py`` to ``timm/models`` folder in repo ``https://github.com/huggingface/pytorch-image-models`` to use it. 

## Inference

``inference`` contains implementations for quantized inference. ``inference/vit`` contains inference implementations for pretrained ViT models on ImageNet. Use ``inference/vit/test_all.py`` to run the evaluations. ``inference/vit/check_bits.py`` computes the average bits per parameter using Huffman Encoding on quantized parameters. 

``inference/llm`` contains implementations for quantized LLM inference. Place the files to ``lm_eval/models`` folder in repo ``https://github.com/EleutherAI/lm-evaluation-harness/tree/main`` to use it. 
