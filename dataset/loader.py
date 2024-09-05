from .dataset_arguements import DatasetConfig, NougatDatasetConfig
from .dummy import DummyDataset
from .dataset import NougatDataset
def load_test_dataset(args: DatasetConfig, branch:str):
    raise NotImplementedError


def load_data(args: NougatDatasetConfig, needed=['train', 'valid'], only_metadata=False, test_dataloader=False):
    
    if args.debug:
        train_dataset = DummyDataset(config = args, length=10000)
        valid_dataset = DummyDataset(config = args, length=300)
        test_dataset  = DummyDataset(config = args, length=300)
        return {'train':train_dataset, 
                'valid':valid_dataset, 
                'test':test_dataset,
                'dev':valid_dataset}
    #tokenizer = args.processor.get_processor().tokenizer
    processor = args.processor.get_processor()
    datasets = {}
    datasets['train']= NougatDataset(args.train_dataset_path, split='train',      processor = processor, config=args, enable_Timer=test_dataloader , dummy=args.debug)
    datasets['valid']= NougatDataset(args.valid_dataset_path, split='validation', processor = processor, config=args, enable_Timer=test_dataloader , dummy=args.debug)
    datasets['test']=None
    return datasets