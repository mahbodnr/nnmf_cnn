from load_model_data import load_model_data    
import pytorch_lightning as pl

model_path = r"model_checkpoints/vit_l_cifar10_elegs_20240617220939.ckpt"
net, train_dl, test_dl, args = load_model_data(model_path, args={"eval_batch_size": 2})

trainer = pl.Trainer()
res = trainer.test(net, test_dl, verbose=True)
