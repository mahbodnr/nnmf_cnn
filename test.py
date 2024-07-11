from load_model_data import load_model_data    
import pytorch_lightning as pl
trainer = pl.Trainer()

model_path = r"model_checkpoints/vit_l_cifar10_elegs_20240617220939.ckpt"
# model_path = r"model_checkpoints/nnmf_vit_l_cifar10_sfwuk_20240617221726.ckpt"
model_path = r"model_checkpoints/vit_cifar10_xajzt_20240614151727.ckpt"
model_path = r"model_checkpoints/cnn_cnn_top_cifar10_vihkj_20240620095212.ckpt"
model_path = r"model_checkpoints/vit_cifar10_jwuvn_20240620105728.ckpt"
net, train_dl, test_dl, args = load_model_data(model_path, args={"eval_batch_size": 128})
print(net)
res = trainer.test(net, test_dl, verbose=True)

net, train_dl, test_dl, args = load_model_data(model_path, args={"eval_batch_size": 2})
res = trainer.test(net, test_dl, verbose=True)
