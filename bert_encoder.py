from imports import *
from transformers import AutoModel

class BertEncoder(nn.Module):
    
    def __init__(self):
        """
        :param lang: str, train bert encoder for a given language
        """
        super(BertEncoder, self).__init__()
        
        self.bert = AutoModel.from_pretrained("Twitter/twhin-bert-base")
        
        self.feature_size = self.bert.config.hidden_size
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        print("Device", self.device)

        self.bert.to(self.device)
        
    def forward(self, input_ids):
        """
        :param input_ids: list[str], list of tokenised sentences
        :return: last hidden representation, torch.tensor of shape (batch_size, seq_length, hidden_dim)
        """
        input_ids = input_ids.to(self.device) # move to GPU if available
        if int((transformers.__version__)[0]) >= 4:
            last_hidden_state = self.bert(input_ids=input_ids).last_hidden_state
        else:
            last_hidden_state, pooler_output = self.bert(input_ids=input_ids)
        return last_hidden_state