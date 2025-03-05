import torch
import torch.nn  as nn 
import torch.nn.functional as F 

from config import Config

from MyLLM.myllm.model.model import Transformer

# building a unified structure for transofrmer decodeer deinfifntion

class GPT:
    def __init__(self, config: Config):
        self.config = config
        self.model = Transformer(config)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.lm_head.weight = self.model.transformer.wte.weight

        self.apply(self._init_weights)
        self.model.apply(self._init_weights)
        self.lm_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = self.model(x)
        x = self.lm_head(x)
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        self.log("predict_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None):
        return self.model.generate(idx, max_new_tokens, temperature, top_k, top_p)
    
    def save(self, path):
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
        return self
    
    def to(self, device):
        self.to(device)
        return self
    
    def __call__(self, x):
        return self.forward(x) 
    


