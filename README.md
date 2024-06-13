## ConvNeXt tests:

#### ConvNeXt
```
python3 run.py --wandb --model-name=convnext --lr=0.001 --lr-scheduler=cosine --min-lr=0 --david-loader --max-epochs=500 --criterion=msece  --batch-size=512
```

#### NNMF ConvNeXt
```
python3 run.py --wandb --model-name=nnmf_convnext --lr=0.001 --lr-scheduler=cosine --min-lr=0 --david-loader --max-epochs=500 --criterion=msece  --batch-size=512 
```