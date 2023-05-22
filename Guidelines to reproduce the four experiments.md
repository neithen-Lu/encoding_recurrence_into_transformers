# Guidelines to reproduce the four experiments 

## The REM module

Our RSA module is basically a weighted average of the standard self-attention and $\mathbf{P}\mathbf{V}$  where $\mathbf{P}$ is the REM. So the most important part is to produce the REM $\mathbf{P}$ given the parameters $(\eta,\nu,\theta)$. We then present the pseudo code for the corresponding REM module that does this job.

```
class REM(k1,k2,k3,k4,k5,k6,d,truncation):
		# initialize the object
    ...
    def get_sinusoid(L,theta):
        M = Hadamard_product(L,theta)
        s1 = cos(M[:k2, ])
        s2 = sin(M[k2:(k2+k3), ])
        s3 = cos(M[(k2+k3):(k2+k3+k4), ])
        s4 = sin(M[(k2+k3+k4):, ])
        s = concat(s1,s2,s3,s4)
        return s
		
		def forward(eta,nu,theta):
				lambda = tanh(eta)
        gamma = sigmoid(nv)
        L = create_Toeplitz_3D(d,truncation) # L is of shape (n_heads x query_len x key_len)
        s1,s2,s3,s4 = get_sinusoid(L,theta)
        powered_lambda = pow(lambda,L)
        powered_gamma = pow(gamma,L)
        REM = concat(powered_lambda,Hadamard_product(powered_gamma,s))
        return REM
```

The gated mechanism is implemented by defining a trainable parameter *mu* and modify the attention map by

```
attn = (1-sigmoid(mu)) * softmax(attn_score) + sigmoid(mu) * REM
```

## Time series

Most of the codes are modified from the [Informer repo](https://github.com/zhouhaoyi/Informer2020). The Transformer-XL model is modified from [transformer-xl repo](https://github.com/kimiyoung/transformer-xl), and the logsparse transformer is constructed by ourselves, with structure very similar to Informer, just change the way it selects the query. All models are incorporated with the time-feature embedding in Informer.

### Change on data sampling scheme

The data sampling scheme is replaced with a sequential sampler, because Transformer-XL needs this sequential sampling to build recurrence between batches.

### Change on Informer

##### Change on attention

We modify the three modules in attn.py into RSA version. Specifically, in ProbAttention module, the context is updated with the RSA module.

##### Change on model structure

The distilling operation in informer.py is removed.

## Formal language

The codes are modified form the [Transformer-Formal-Languages repo](https://github.com/satwik77/Transformer-Formal-Languages).

## Code language

The codes are modified form the [CodeT5 repo](https://github.com/salesforce/CodeT5).

In this task, the modification is slightly more complicated because we use the pretrained weights from CodeT5 instead of training from scratch. So the procedure is to first replace the original attention module from the model, and then load the weights:

```
model = load_pretrained(path)
state_copy = model.state_dict()

for i from 1 to encoder_layers:
	 model.encoder.block[i].layer[0].SelfAttention = T5SelfAttention_RSA(args)
for i from 1 to decoder_layers:
	 model.decoder.block[i].layer[0].SelfAttention = T5SelfAttention_RSA(args)
	 
model.load_state_dict(state_copy,strict=False)
```

## Natural language

Most of the codes are modified from the [Nvidia Transformer-XL repo](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/Transformer-XL).

### Change on BRT

As there is no official pytorch implementation of BRT, we modify the BRT model based on this [repo](https://github.com/dashstander/block-recurrent-transformer). The model is slightly different from BRT's paper (see the references therein). Followed by the BRT's paper, we insert one block recurrent layer into the Transformer-XL model.