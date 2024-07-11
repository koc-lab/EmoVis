from imports import *
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.preprocessor import TextPreProcessor

class TwitterData(Dataset):
    def __init__(self, max_length=128, split='train'): 
        # args: dictionary of arguments
        # filename: name of the file to be loaded
        # max_length: maximum length of the input sequence
        # dataset: dataset to be loaded
        # bert_tokeniser: bert tokeniser to be used
        self.max_length = max_length
        self.split = split
        self.dataset = self.load_dataset()
        self.bert_tokeniser = AutoTokenizer.from_pretrained("Twitter/twhin-bert-base") 
        self.preprocessed_dataset = self.process_data()


    def load_dataset(self):
        """
        returns the dataset after being preprocessed and tokenised
        """
        print("Downloading Dataset {}...".format(''))
        dataset = load_dataset('sem_eval_2018_task_1', 'subtask5.english', split=self.split) # change for the dataset
        return dataset

    def process_data(self):
        preprocessor = self.Preprocessor()
        
        
        Question = "ira anticipación asco miedo alegría amor optimismo pesimismo tristeza sorpresa or confianza?"
        label_names = ['ira', 'anticipación', 'asco', 'miedo', 'alegría', 'amor', 'optimismo',
                           'pesim', 'tristeza', 'sorpresa', 'confianza']

  
        inputs, lengths, label_indices, sentence_tokens = [], [], [], []
        for row in tqdm(self.dataset, desc="Preprocessing Dataset for {} {}...".format(str(self.split),'')):
            x = ' '.join(preprocessor(row['Tweet']))
            x = self.bert_tokeniser(x,  
                                    add_special_tokens=True,
                                    max_length=self.max_length,
                                    pad_to_max_length=True,
                                    truncation=True)
            input_id = x['input_ids']
            input_length = len([i for i in x['attention_mask'] if i == 1])
            inputs.append(input_id)
            lengths.append(input_length)

            #convert input ids to tokens in a form similar to ['[CLS]', 'hello', ',', 'my', 'dog', 'is', 'cute', '[SEP]']
            sentence_tokens.append(self.bert_tokeniser.convert_ids_to_tokens(input_id))

        preprocessed_dataset = {'input': torch.tensor(inputs, dtype=torch.long), 
                                            'length': torch.tensor(lengths, dtype=torch.long), 
                                            'sentence_tokens': sentence_tokens}

        return preprocessed_dataset
    
    def Preprocessor(self):
        # Preprocess the text
        preprocessor = TextPreProcessor(
            normalize=['url', 'email', 'phone', 'user', 'time'],
            annotate={"hashtag", "elongated", "allcaps", "repeated", 'emphasis', 'censored'},
            all_caps_tag="wrap",
            fix_text=False,
            segmenter="twitter_2018",
            corrector="twitter_2018",
            unpack_hashtags=True,
            unpack_contractions=True,
            spell_correct_elong=False,
            tokenizer=SocialTokenizer(lowercase=True).tokenize).pre_process_doc
        return preprocessor

    def __getitem__(self, index):
        inputs = self.preprocessed_dataset['input'][index]
        length = self.preprocessed_dataset['length'][index]
        return inputs, length
 
    def __len__(self):
        return len(self.preprocessed_dataset['input'])