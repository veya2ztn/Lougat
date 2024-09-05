from PIL import Image,ImageOps,ImageDraw,ImageColor
import torch
import os
import json
import subprocess
from pathlib import Path
import json

def visual_box(png_path,boxes,save_path,color=(255,0,0),image_size = [672,896],texts=None,fill=True):
    img = Image.open(png_path).resize(image_size)
    img=img.convert('RGBA')
    transp = Image.new('RGBA', image_size, (0,0,0,0))
    draw = ImageDraw.Draw(transp, "RGBA")   
    # draw = ImageDraw.Draw(img)
    boxes = boxes.reshape(-1,2,2)
    if not isinstance(color,tuple): # 'red'
        color = ImageColor.getrgb(color)
    if fill:
        fill_color = color + (80,)   # 一半的透明度
    else:
        fill_color = (255,255,255,0)    # 透明
    
    try:
        for i,box in enumerate(boxes):
            if box[1][0] < box[0][0]: 
                print('x2<x1')
                box[1][0] = box[0][0]
            if box[1][1] < box[0][1]:
                print('y2<y1')
                box[1][1] = box[0][1]
            resized_box = torch.empty_like(box)
            resized_box[:,0] = box[:,0]*image_size[0]   # x
            resized_box[:,1] = box[:,1]*image_size[1]   # y
            
            
            
            draw.rectangle([tuple(resized_box[0]),tuple(resized_box[1])],outline=color,fill=fill_color)
            if texts:
                if color == 'blue':
                    draw.text((resized_box[0][0]-10,resized_box[0][1]-20),texts[i].encode("utf-8").decode("latin1"),fill=color)
                else:
                    draw.text((resized_box[0][0]-10,resized_box[0][1]-10),texts[i].encode("utf-8").decode("latin1"),fill=color)
        img.paste(Image.alpha_composite(img, transp))
        img.save(save_path)   
    except Exception as e:
        print(e)
        
def is_blank_box(image, box, white_thres=220, image_size = [672,896],ratio_thres=0.75):
    [[x1,y1],[x2,y2]] = box
    x1,y1,x2,y2 = int(x1*image_size[0]),int(y1*image_size[1]),int(x2*image_size[0]),int(y2*image_size[1])
    white = 0
    for y in range(max(y1,0), min(y2,image_size[1])):
        for x in range(max(x1,0), min(x2,image_size[0])):
            pixel_value = image.getpixel((x, y))
            if all(value > white_thres for value in pixel_value):
                white += 1
    white_ratio =  white/((x2 - x1) * (y2 - y1))
    # return white_ratio
    if white_ratio < ratio_thres:
        return False    # 非空白框
    else:
        return True  # 空白

def interact_with_human(prompt_pred,flask_png_path,save_path,color ='red',image_size = [672,896]):
    # cross_attn_weights = torch.stack(outputs.cross_attentions)[:,:,:,-1:,:] # [4, bs,16,1,588]
    # 在flask目录下保存图片
    
    # 读取用户输入
    img = Image.open(flask_png_path).resize(image_size)
    draw = ImageDraw.Draw(img)
    prompt_pred[:,0] *= image_size[0]
    prompt_pred[:,1] *= image_size[1]
    if prompt_pred[0,0] < prompt_pred[1,0] and prompt_pred[0,1] < prompt_pred[1,1]:
        draw.rectangle([tuple(prompt_pred[0]),tuple(prompt_pred[1])],outline=color)
    # draw.text((prompt_pred[0][0]-10,prompt_pred[0][1]-20),f'{prompt_pred.tolist()}',fill='black')
    img.save(flask_png_path)     
    
    os.chdir('flask-image-annotator')
    process = subprocess.Popen(['python','app.py'])
    os.chdir('../')
    process.terminate()
    with open('flask-image-annotator/out.json','r') as fi:
        user_input = fi.readline()
    if user_input:  # 不画框直接默认不变
        dct = json.loads(user_input)
        token_user = dct['name'].replace('\\n','\n')  # 将\\n恢复为\n，但保留Unicode字符串
        prompt_user = [[dct['x1']/image_size[0],dct['y1']/image_size[1]],[dct['x2']/image_size[0],dct['y2']/image_size[1]]]
        visual_box(png_path=save_path,boxes=torch.tensor(prompt_user),save_path=save_path,color='blue',image_size = [672,896],fill=True)      # red: 正确人工框
        return prompt_user,token_user
    else:
        return [],''
    
if __name__ == '__main__':
    with open('data/arxiv_train_data/table_train_color_black013.jsonl') as fi:
        lines = fi.readlines()
    save_path = 'data/case/position_data_demo/tmp.png'
    for line in lines[500:]:
        dct = json.loads(line)
        pretexts = dct['pretext']
        prompts = dct['prompt']
        png_path = f"data/arxiv_train_data/{dct['image']}"
        box_txt = [p for p in prompts if p != [[-1,-1],[-1,-1]]]
        box_math,box_table = [],[]
        idx = 0
        while idx < len(pretexts):
            # '\begin{table}' 
            if idx < len(pretexts) and '\\begin{table}' in pretexts[idx]:
                # '\end{table}'
                while idx < len(pretexts) and not '\\end{table}' in pretexts[idx]:
                    if prompts[idx] != [[-1,-1],[-1,-1]]:
                        box_txt.remove(prompts[idx])
                        box_table.append(prompts[idx])
                    idx += 1
            #  '\(' 
            if idx < len(pretexts) and  '\(' in pretexts[idx]:
                while idx < len(pretexts) and not '\)' in pretexts[idx]:     # '\)' 
                    if prompts[idx] != [[-1,-1],[-1,-1]]:
                        box_txt.remove(prompts[idx])
                        box_math.append(prompts[idx])
                    idx += 1
            # '\['
            if idx < len(pretexts) and '\[' in pretexts[idx]:
                while idx < len(pretexts) and not '\]' in pretexts[idx]:   # '\]'
                    if prompts[idx] != [[-1,-1],[-1,-1]]:
                        box_txt.remove(prompts[idx])
                        box_math.append(prompts[idx])
                    idx += 1
            idx += 1
        if box_table:#len(box_math)>100 or 
            visual_box(png_path,torch.tensor(box_txt),save_path,color=(200, 115, 190),image_size = [672,896],texts=None) # pink
            visual_box(save_path,torch.tensor(box_math),save_path,color=(70, 80, 180),image_size = [672,896],texts=None)   # blue
            visual_box(save_path,torch.tensor(box_table[1:-1]),save_path,color=(40,140,60),image_size = [672,896],texts=None) # green