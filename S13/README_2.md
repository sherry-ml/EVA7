# Session 13

This ReadMe file describes the code of different classes in short as provided in https://colab.research.google.com/drive/1zLMtRQt5k9dTgZP42mGhRnjs0QF-T5PG?usp=sharing#scrollTo=hS3u4DAaBSoo

It is a badly written code with huge scope for improvement and not suitable for gaining understanding by a starter in transformers.

1) Class PatchEmbeddings takes the input data and breaks it into 196 patches of 16x16x3 images each of which is flattened to produce an output of 196 patches X 768 embedding layer
2) Class ViTEmbeddings passes the input data through PatchEmbedding object to get patch embeddings in shape - batch_size*196*768. Next it creates class token of size 32x1x768 and concatenates it with patch embeddings to produce an output of batch_size*197*768. Next, it adds position embedding to each of the 197 tokens and then passes the output through dropout object.
3) Class ViTConfig sets the different configuration parameters which are passed to different class objects like size of patch, image size, dropout probability etc
4) Class ViTSelfAttention is taking output returned by ViTEmbeddings class object , passes it through query, key and value linear layers and then breaks resulting 768 unit embedding layer into 12*64 and through different transposition operations it is converted into shape of 197x12*64. (Converting 768 into 12*64 indicates that each patch is broken into 12 pieces of patches each with embedding size of 64 each of which is passed through different attention head. There are 12 heads and each head consists of a query, key and value layer. Final output of each attention head is concatenated to obtain final output of self attention layer.)  Matrix multiplication is then applied on resulting output of query and key layers to obtain attention scores which is further divided by square root of attention head size (64**0.5 = 8). Softmax operation is then applied on top of resulting output followed by dropout operation. Matrix multiplication is then applied between resulting output and value linear layer output to get output of self attention block. Final output of the self attention block has to be of the same dimension as what we got after embedding operation. In this case it should be batch_size*197*768
5) Class ViTSelfOutput takes output ViTSelfAttention object and passes it through a linear layer plus dropout operation
6) Class ViTAttention combines the outputs from ViTSelfAttention and ViTSelfOutput objects.

<<More to follow>>
