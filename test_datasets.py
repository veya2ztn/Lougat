from dataset.dataset_arguements import UparxiveDatasetConfig,ParquetUpaxiveConfig,ParquetUpaxive2kConfig
from model.processor_arguements import UparxiveBetaProcessorConfig,Uparxive1KProcessorConfig,UparxiveProcessorConfig,UparxiveProcessor2kConfig
from PIL import Image, ImageDraw
import numpy as np

processor_config = Uparxive1KProcessorConfig()

dataset_config = ParquetUpaxiveConfig(root_name='data')
dataset_config._processor = processor_config
dataset_config._max_length = 4096
dataset_config._decoder_start_token_id  = 0

from dataset.loader import load_data
datasets = load_data(dataset_config)
dataset = datasets['train']
tokenizer = dataset.processor.tokenizer
dataset.train()
idx=30
for idx in range(20,30):
    input_tensor, token_ids, token_types, prompts, attention_mask = dataset[idx]
    image_data = input_tensor.permute(1,2,0).numpy()
    image_data = image_data*np.array([[datasets['train'].processor.image_processor.image_std]])
    image_data = image_data+np.array([[datasets['train'].processor.image_processor.image_mean]])
    image_data = np.round(image_data*255).astype('uint8')
    image = Image.fromarray(image_data)

    bboxes = prompts[attention_mask].view(-1,4)[1:]
    token_ids=token_ids[attention_mask]
    
    token_string = tokenizer.batch_decode(token_ids[None], skip_special_tokens=False)[0]
    
    width, height= image.size
    print(width, height)
    bboxes[:,[0,2]]*=width
    bboxes[:,[1,3]]*=height


    draw = ImageDraw.Draw(image)

    # Draw bounding boxes
    for box in bboxes:
        # Unpack the bounding box
        x_min, y_min, x_max, y_max = box
        # Draw the rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline='red')
    image.save(f"images/test.snap1.{idx}.png")
    with open(f"images/text.snap.{idx}.txt",'w') as f:
        f.write(token_string)
exit()

dataset.eval()
dataset.force_use_augmentation = False

input_tensor, token_ids, token_types, prompts, attention_mask = dataset[idx]
image_data = input_tensor.permute(1,2,0).numpy()
image_data = image_data*np.array([[datasets['train'].processor.image_processor.image_std]])
image_data = image_data+np.array([[datasets['train'].processor.image_processor.image_mean]])
image_data = np.round(image_data*255).astype('uint8')

image = Image.fromarray(image_data)

bboxes = prompts[attention_mask].view(-1,4)[1:]
width, height= image.size
bboxes[:,[0,2]]*=width
bboxes[:,[1,3]]*=height
print(width, height)

draw = ImageDraw.Draw(image)

# Draw bounding boxes
for box in bboxes:
    # Unpack the bounding box
    x_min, y_min, x_max, y_max = box
    # Draw the rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red')
image.save("test.snap2.png")

dataset.eval()
dataset.force_use_augmentation = True


input_tensor, token_ids, token_types, prompts, attention_mask = dataset[idx]
image_data = input_tensor.permute(1,2,0).numpy()
image_data = image_data*np.array([[datasets['train'].processor.image_processor.image_std]])
image_data = image_data+np.array([[datasets['train'].processor.image_processor.image_mean]])
image_data = np.round(image_data*255).astype('uint8')

image = Image.fromarray(image_data)

bboxes = prompts[attention_mask].view(-1,4)[1:]
width, height= image.size
bboxes[:,[0,2]]*=width
bboxes[:,[1,3]]*=height
print(width, height)

draw = ImageDraw.Draw(image)

# Draw bounding boxes
for box in bboxes:
    # Unpack the bounding box
    x_min, y_min, x_max, y_max = box
    # Draw the rectangle
    draw.rectangle([x_min, y_min, x_max, y_max], outline='red')
image.save("test.snap3.png")