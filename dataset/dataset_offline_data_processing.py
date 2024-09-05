import pandas as pd
import os
import fitz
fitz.TOOLS.mupdf_display_errors(on=False)
from PIL import Image, ImageOps
import io
from tqdm.auto import tqdm
from multiprocessing import Pool
import sys
from dataset.resource_utils import robust_csv_reader,filter_df, obtain_cleaned_token_type_bbox_pair,obtain_clean_tokenid_bbox_pair
from transformers import AutoTokenizer
import json
import numpy as np
ROOT="data"
csvpath = sys.argv[1]
if csvpath.endswith('.csv'):
    path2name = {}
    csv_file_list = []
    csvpath = sys.argv[1]
    #records = pd.read_csv("data/archive_tex.colorful.csv.fold/archive_tex.colorful.nobbl.perfect_successed.pdf_box_pair.csv")
    records = pd.read_csv(csvpath)
    for arxiv_path, pdf_name,page_id in records[['arxivpath', 'pdf_name', "page_id"]].values:
        if arxiv_path not in path2name:
            path2name[arxiv_path] = pdf_name
        csv_file_path = os.path.join(ROOT, arxiv_path, 'boxed_pdf_image', 'text_bbox_json',f"page_{page_id}.json")
        csv_file_list.append(csv_file_path)

    pdf_file_path_list = []
    for arxiv_path, pdf_name in path2name.items():
        pdf_file_path = os.path.join(ROOT, arxiv_path, "boxed_pdf_image", pdf_name+'.colorful.text_only.pdf')
        pdf_file_path_list.append(pdf_file_path)
elif csvpath.endswith('.folds.filelist'):
    with open(csvpath,'r') as f: 
        csv_file_list= [line.strip() for line in f]
elif csvpath.endswith('.filelist'):
    with open(csvpath,'r') as f: 
        foldpathlist= [line.strip() for line in f]
    csv_file_list = []
    for foldpath in foldpathlist:
        csvfold = os.path.join(foldpath, 'boxed_pdf_image', 'text_bbox_json')
        csvpaths= os.listdir(csvfold)
        csvpaths= [os.path.join(csvfold, t) for t in csvpaths]
        csv_file_list.extend(csvpaths)
print(f"should deal with {len(csv_file_list)} items")

dpi=150
def offline_one_pdf(pdf_file_path):
    pdf_root = os.path.dirname(pdf_file_path)
    with fitz.open(pdf_file_path) as pdf:
        for page_idx in range(len(pdf)):
            #old_img_save_path = pdf_file_path.replace('.colorful.text_only.pdf', f'.page_{page_idx}.png').replace('boxed_pdf_image/', f'boxed_pdf_image/images')
            img_save_path = pdf_file_path.replace('.colorful.text_only.pdf', f'.page_{page_idx}.png').replace('boxed_pdf_image/', f'boxed_pdf_image/images/')
            if os.path.exists(img_save_path):continue
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            #os.rename(old_img_save_path, img_save_path)
            page = pdf[page_idx]
            image = Image.open(io.BytesIO(page.get_pixmap(colorspace=fitz.csGRAY,dpi=dpi).pil_tobytes(format="PNG")))
            gray_image = ImageOps.grayscale(image)
            #img   = ImageOps.autocontrast(gray_image, cutoff=1)
            img = gray_image.point(lambda p: 255 if p == 255 else 0) ### <-- this one let all no-white color become black
            img.save(img_save_path)


tokenizer = AutoTokenizer.from_pretrained('config/processor/uparxive')
tokenizer_key = tokenizer.name_or_path.split("/")[-1]
def offline_one_csv(csv_file_path):
    
    try:
        token_path = csv_file_path.replace('text_bbox_json', f'text_json_cache').replace('.json',f'.{tokenizer_key}.json')
        if os.path.exists(token_path):
            # with open(token_path,'r') as f:
            #     data = json.load(f)
            return csv_file_path, "Pass"
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        df = robust_csv_reader(csv_file_path)
        status, df = filter_df(df)
        if not status:
            return csv_file_path, df
        
        formated_token_bbox_pair = obtain_cleaned_token_type_bbox_pair(df)
        pretext, token_ids, bboxes, text_type = obtain_clean_tokenid_bbox_pair(formated_token_bbox_pair,tokenizer)
        if sum([len(t) for t in token_ids])>4096:
            return csv_file_path, "Too long sequence"
        with open(token_path,'w') as f:
            json.dump({
                "token_ids":token_ids,
                "bboxes":bboxes,
                "text_type":text_type
            }, f)
        return csv_file_path, "Pass"
    except:
        tqdm.write(csv_file_path)
        return csv_file_path, "Error"


num_processes = 100
if num_processes > 1:
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(offline_one_csv, csv_file_list), total=len(csv_file_list)))
else:
    results= []
    for csv_file_path in tqdm(csv_file_list):
        results.append(offline_one_csv(csv_file_path))
        
reason_pool = {}
for path, reason in results:
    if reason not in reason_pool:
        reason_pool[reason] = []
    reason_pool[reason].append(path)

for reason, pathlist in reason_pool.items():
    print(f"{reason}=>{len(pathlist)}")

dataframe_row = []
for csvpath in tqdm(reason_pool['Pass']):
    
    #  archive_tex.colorful.addbbl.partial_successed/2110/2110.00727/boxed_pdf_image/text_bbox_json/page_0.csv
    #->archive_tex.colorful.addbbl.partial_successed/2110/2110.00727/boxed_pdf_image/text_bbox_json/
    #->archive_tex.colorful.addbbl.partial_successed/2110/2110.00727/boxed_pdf_image/
    foldpath= os.path.dirname(os.path.dirname(csvpath))
    get_pdf_name = [t for t in os.listdir(foldpath) if t.endswith('.colorful.pdf')]
    if len(get_pdf_name) == 0:
        print(foldpath)
        continue
    pdf_name = get_pdf_name[0].replace('.coloful.pdf','')
    page_id  = int(os.path.basename(csvpath).replace('page_','').replace('.json',''))
    relpath  = os.path.relpath(foldpath, "/nvme/zhangtianning/datasets/whole_arxiv_data/whole_arxiv_all_files")
    arxivpath= os.path.dirname(relpath)


    split    = np.random.choice(['Train','Test','Valid'],p=[0.8,0.1,0.1])
    dataframe_row.append([arxivpath,pdf_name,page_id,split])

good_data_csv = pd.DataFrame(dataframe_row,columns=["arxivpath","pdf_name","page_id","split"])
good_data_csv_path = sys.argv[1].replace('.filelist','.pdf_box_pair')+'.csv'
good_data_csv.to_csv(good_data_csv_path)

dataframe_row = []
for reason, pathlist in reason_pool.items():
    if reason in ['Pass']:continue
    for csvpath in tqdm(reason_pool[reason]):
        
        #  archive_tex.colorful.addbbl.partial_successed/2110/2110.00727/boxed_pdf_image/text_bbox_json/page_0.csv
        #->archive_tex.colorful.addbbl.partial_successed/2110/2110.00727/boxed_pdf_image/text_bbox_json/
        #->archive_tex.colorful.addbbl.partial_successed/2110/2110.00727/boxed_pdf_image/
        foldpath= os.path.dirname(os.path.dirname(csvpath))
        get_pdf_name = [t for t in os.listdir(foldpath) if t.endswith('.colorful.pdf')]
        if len(get_pdf_name) == 0:
            print(foldpath)
            continue
        pdf_name = get_pdf_name[0].replace('.coloful.pdf','')
        page_id  = int(os.path.basename(csvpath).replace('page_','').replace('.json',''))
        relpath  = os.path.relpath(foldpath, "/nvme/zhangtianning/datasets/whole_arxiv_data/whole_arxiv_all_files")
        arxivpath= os.path.dirname(relpath)
        split    = np.random.choice(['Train','Test','Valid'],p=[0.8,0.1,0.1])
        dataframe_row.append([arxivpath,pdf_name,page_id,split, reason])
bad_data_csv = pd.DataFrame(dataframe_row,columns=["arxivpath","pdf_name","page_id","split","fail_reason"])
bad_data_csv_path = sys.argv[1].replace('.filelist','.bad_pdf_box_pair')+'.csv'
bad_data_csv.to_csv(bad_data_csv_path)

    
# for reason, pathlist in reason_pool.items():
#     with open(f"data/archive_tex.colorful.csv.fold/analysis/offline_data_processing_{reason}.txt", 'w') as f:
#         for path in pathlist:
#             f.write(path+'\n')





# print(csv_file_list[0])
# offline_one_csv(csv_file_list[0])

# csv_file_list = [
# "data/archive_tex.colorful.nobbl.perfect_successed/1908/1908.06000/boxed_pdf_image/text_bbox_json/page_18.csv",
# "data/archive_tex.colorful.nobbl.perfect_successed/1908/1908.06000/boxed_pdf_image/text_bbox_json/page_16.csv",
# "data/archive_tex.colorful.nobbl.perfect_successed/1908/1908.06000/boxed_pdf_image/text_bbox_json/page_21.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_8.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_15.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_17.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.06140/boxed_pdf_image/text_bbox_json/page_2.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_14.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_13.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_5.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_12.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.04641/boxed_pdf_image/text_bbox_json/page_3.csv",
# "data/archive_tex.colorful.addbbl.perfect_successed/2108/2108.05168/boxed_pdf_image/text_bbox_json/page_16.csv",
# "data/archive_tex.colorful.nobbl.partial_successed/1909/1909.12725/boxed_pdf_image/text_bbox_json/page_4.csv",
# "data/archive_tex.colorful.nobbl.partial_successed/1909/1909.12725/boxed_pdf_image/text_bbox_json/page_6.csv",]

# for csv_file_path in tqdm(csv_file_list):
#     offline_one_csv(csv_file_path)