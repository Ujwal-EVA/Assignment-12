
# Changes Made to Model Code

## 1. Defined Missing Variables

- **max_length**: The variable `max_length = 50` just before the generation loop to specify the length of the generated sequence.

## 2. DataLoaderLite Initialization
- In the original code, `train_loader` was referenced but never initialized, causing an error. Added the `DataLoaderLite` class initialization:
  
  train_loader = DataLoaderLite(B=num_return_sequences, T=32)
  

## 3. Model Training Loop Adjustments
- The training loop was modified for handling of gradients. 
- The optimizer and scheduler were initialized as follows:
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
  scheduler = CosineAnnealingLR(optimizer, T_max=50)
  

## 4. Sequence Generation Loop
- The generation loop was fixed by initializing the input sequence `x` with random tokens:
  
  x = torch.randint(0, model.config.vocab_size, (num_return_sequences, 1), device=device)
  
- The loop now continues to generate tokens until `max_length` is reached. It uses the `softmax` function and `topk` to sample the next token based on probability distribution.

## 5. Device Compatibility
- The code now dynamically selects the device (`cpu`, `cuda`, or `mps`) for training based on the available hardware:
  
  device = 'cpu'
  if torch.cuda.is_available():
      device = 'cuda'
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
      device = "mps"
 

## 6. Model Weights Initialization
- A more robust weight initialization approach was introduced for layers such as `Linear` and `Embedding` using the `_init_weights` method.
