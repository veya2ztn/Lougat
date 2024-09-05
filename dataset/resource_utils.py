import numpy as np
from tqdm.auto import tqdm
import numpy as np
NONE_BBOX_CASE=[None, np.nan,[[0,0],[0,0]],[[-1,-1],[-1,-1]]]

def is_good_a_math_sequence(block_math_indices):
    block_math_indices = np.array(block_math_indices)
    if len(block_math_indices) % 2 == 0:
        if all((block_math_indices[1::2] - block_math_indices[::2]) == 2):
            return 'good'
        else:
            return None
    else:
        if is_good_a_math_sequence(block_math_indices[1:]):
            return 'left'
        elif is_good_a_math_sequence(block_math_indices[:-1]):
            return 'right'
        else:
            return None

def good_box(box):
    if box in NONE_BBOX_CASE:return None
    if isinstance(box,str):box= eval(box)
    assert isinstance(box, list)
    assert isinstance(box[0], list)
    return box    

def fill_empty_bbox_follow_last(bbox_ordered):
    boxes = []
    for t in bbox_ordered:
        box = good_box(t) 
        if box is None:
            if len(boxes)>0:
                ((x0, y0),(x1,y1))= boxes[-1]
                box = [[x1,y0],[x1+0.001,y1]]
            else:
                box  = [[0,0],[0.001,0.001]]
        boxes.append(box)
    return boxes

def fast_fix_brace_closure(df):
    block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()
    if len(block_math_indices)==0:return df
    start_indice, end_indice = block_math_indices[0], block_math_indices[-1]
    if df.iloc[start_indice].markdown == '\n\]\n':
        if start_indice == 1:
            new_row = pd.DataFrame({'text_type': ['<block_math>'], 'markdown': ['\n\[\n'], 'pdf': [''] ,'status': 'invisable','bbox': None})
            df = pd.concat([new_row, df]).reset_index(drop=True)
            block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()
        elif start_indice == 0:
            df = df.drop(index=[start_indice]).reset_index(drop=True)
            block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()
    if len(block_math_indices)==0:return df
    start_indice, end_indice = block_math_indices[0], block_math_indices[-1] 
    if df.iloc[end_indice].markdown == '\n\[\n':
        if end_indice == len(df)-2:
            new_row = pd.DataFrame({'text_type': ['<block_math>'], 'markdown': ['\n\]\n'], 'pdf': [''] ,'status': 'invisable','bbox': None})
            df = pd.concat([df,new_row]).reset_index(drop=True)
            block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()

        elif end_indice == len(df)-1:
            df = df.drop(index=[end_indice]).reset_index(drop=True)
            block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()
    if len(block_math_indices)==0:return df
    start_indice, end_indice = block_math_indices[0], block_math_indices[-1] 
    return df

def filter_df(df):
    df = df[~((df['text_type']=='<text>') & (df['markdown'].str.strip()!=df['pdf'].str.strip())& ~(df['markdown'].str.startswith("#")))]
    df = df[~df['markdown'].isna()]
    df.reset_index(drop=True, inplace=True)
    first_no_nan = df['bbox'].first_valid_index()
    if len(df) > 1.5*df['bbox'].count() or len(df) > 2000 or df['bbox'].count()<100:
        #bad_pdf_bbox_data_path_pair.append([arxivpath, pdf_name, page_id, 'bad_alignment'])
        return False, 'tool_less_bbox'
    
    df = df.loc[first_no_nan-1:]
    df.reset_index(drop=True, inplace=True)
    first_no_nan = df['bbox'].first_valid_index()
    if len(df[df['status']!='invisable'])<100:
        return False, "Too Short sequence"
    df = fast_fix_brace_closure(df)
    block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()
    if len(block_math_indices)>0:
        block_status = is_good_a_math_sequence(block_math_indices)
        #print(block_status)
        markdown_postion ="markdown" # df.columns.get_loc("markdown") #"markdown" #df.columns.get_loc("markdown")
        text_type_postion="text_type"# df.columns.get_loc("text_type")
        if block_status == 'good':
            pass
        elif block_status == 'left':
            if block_math_indices[0] == 1:
                ### lets add a new block_math_indices at the begin
                new_row = pd.DataFrame({'text_type': ['<block_math>'], 'markdown': ['\n\[\n'], 'pdf': [''] ,'status': 'invisable','bbox': None})
                df = pd.concat([new_row, df]).reset_index(drop=True)
                block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()
            else:
                ## then we delete it 
                df.loc[block_math_indices[0], markdown_postion] = None
                block_math_indices = block_math_indices[1:]
        elif block_status == 'right':
            if block_math_indices[-1] == len(df)-2:
                ### lets add a new block_math_indices at the end
                new_row = pd.DataFrame({'text_type': ['<block_math>'], 'markdown': ['\n\]\n'], 'pdf': [''] ,'status': 'invisable','bbox': None})
                df = pd.concat([df,new_row]).reset_index(drop=True)
                block_math_indices = df.index[df['text_type'] == '<block_math>'].tolist()
            else:
                ### then we will romove it 
                df.loc[block_math_indices[-1], markdown_postion] = None
                block_math_indices = block_math_indices[:-1]
        else:
            #bad_pdf_bbox_data_path_pair.append([arxivpath, pdf_name, page_id, 'bad_block_math'])
            return False, 'bad_block_math_match'


        for i in range(0, len(block_math_indices) - 1, 2):
            start = block_math_indices[i]
            end = block_math_indices[i + 1]

            # Check if the closing <block_math> has only one text in it
            if end - start > 2:
                #bad_pdf_bbox_data_path_pair.append([arxivpath, pdf_name, page_id, 'bad_block_math'])
                return False, 'bad_block_math_detected'
            if end - start  == 2:

                df.loc[start + 1:end-1, markdown_postion] = "<BLOCK_MATH>" #'<block_equation>'
                df.loc[start + 1:end-1, text_type_postion] = "<BLOCK_MATH>" #'<block_equation>'
                df.drop(index=[start,end], inplace=True) ## remove the <\n\[\n> and <\n\]\n>
        df = df[~df['markdown'].isna()]
        df.reset_index(drop=True, inplace=True)
    
    return True, df 

def obtain_cleaned_token_type_bbox_pair(df):
    start_token_mapping ={
        "\\(": "<INLINE_MATH_START>",
        "<cap>":"<CAP_START>",
        "#":"#",
        "##":"##",
        "###":"###",
        "####":"####",
        "#####":"#####",
        "######":"######",
        "#######":"#######",
        "########":"########"
    }
    end_token_mapping = {
        "\\)": "<INLINE_MATH_END>",
        "</cap>":"<CAP_END>"
    }
    footnote_tag=["<footnote>"]
    figure_tag = ["<fig>","<figrues>","<figure>"]
    table_tag  = ["<tab>","<tables>","<table>"]
    float_tag  = ["<flt>","<float>","<floats>"]
    all_tag = footnote_tag + figure_tag + table_tag + float_tag
    formated_token_bbox_pair = []
    count = 0 
    for i,(markdown_string, pdf_string, status, text_type, bbox) in df.iterrows():
        assert count == i
        count+=1


        current_text_type = 'special'

        if any(t in markdown_string.lower() for t in all_tag):
            if any(t in markdown_string.lower() for t in footnote_tag):
                special_token = "<FOOTNOTE>"
            elif any(t in markdown_string.lower() for t in figure_tag):
                special_token = "<FIGURE>"
            elif any(t in markdown_string.lower() for t in table_tag):
                special_token = "<TABLE>"
            elif any(t in markdown_string.lower() for t in float_tag):
                special_token = "<FLOAT>"
            else:
                raise NotImplementedError
            start_i = i
            while bbox in NONE_BBOX_CASE and abs(start_i - i) < 10 and start_i < len(df):
                bbox = df.iloc[start_i].bbox
                start_i+=1
                
            formated_token_bbox_pair.append((special_token,bbox,special_token)) 
            continue
        current_text_type = text_type
        if markdown_string.strip() in start_token_mapping:
            start_i = i
            while bbox in NONE_BBOX_CASE and abs(start_i - i) < 10 and start_i < len(df):
                bbox = df.iloc[start_i].bbox
                start_i+=1
                
            special_token = start_token_mapping[markdown_string.strip()]
            formated_token_bbox_pair.append((special_token,bbox,current_text_type)) 
            continue
        if markdown_string.strip() in end_token_mapping:
            start_i = i
            while bbox in NONE_BBOX_CASE and abs(start_i - i) < 10 and start_i >= 0:
                
                bbox = df.iloc[start_i].bbox
                start_i-=1
            special_token = end_token_mapping[markdown_string.strip()]
            formated_token_bbox_pair.append((special_token,bbox,current_text_type)) 
            continue
        ## check should split the <NEWLINE>
        no_right_space_string = markdown_string.rstrip()
        space_left = markdown_string[len(no_right_space_string):]
        if len(space_left) > 0 and "\n" in space_left and text_type == '<text>':
            if len(no_right_space_string)>0:
                formated_token_bbox_pair.append((no_right_space_string,bbox,current_text_type)) 
                # lets modify the bbox to the end of last bbox
                if bbox in NONE_BBOX_CASE:
                    tqdm.write(f"Debug: {markdown_string, pdf_string, status, text_type, bbox}, we remove it as the bbox is nan")
                    continue

                if isinstance(bbox, str):bbox = eval(bbox)

                [x0, y0] = bbox[0]
                [x1, y1] = bbox[1]

                bbox = [[x1,y0],[x1+0.001,y1]]

            formated_token_bbox_pair.append(("<NEWLINE>",bbox,current_text_type)) 
            continue
        start = "" if text_type =="<inline_math>" else " "
        markdown_string = start + markdown_string
        formated_token_bbox_pair.append((markdown_string,bbox,current_text_type))
    return formated_token_bbox_pair

type_mapping = {
            "<FOOTNOTE>" :2, 
            "<FLOAT>":2, 
            "<FIGURE>":2, 
            "<TABLE>":2, 
            "<BLOCK_MATH>":2,
            "<NEWLINE>":0,
            '<inline_math>': 1,
            '<text>': 0,
            'mask' : 0

        }

def obtain_clean_tokenid_bbox_pair(formated_token_bbox_pair,tokenizer):
    pretext   = [t[0] for t in formated_token_bbox_pair]
    token_ids = None
    if tokenizer is not None:
        token_ids = tokenizer(pretext)["input_ids"]
        token_ids = [t[1:-1] for t in token_ids]
    bboxes = []
    for t in formated_token_bbox_pair:
        bbox = t[1]
        if bbox in NONE_BBOX_CASE:
            bbox =None
        elif isinstance(bbox, str):
            if len(bbox.strip())==0:
                bbox = None
            else:
                bbox = eval(bbox)
        else:
            (x0, y0),(x1,y1) = bbox
        bboxes.append(bbox)
    
    text_type = [t[2] if isinstance(t[2],int) else type_mapping[t[2]] for t in formated_token_bbox_pair]
    
    
    return pretext, token_ids, bboxes, text_type
import pandas as pd
import re,csv

def parse_csv_line(line):
    reader = csv.reader([line])
    try:
        csvresult=next(reader)
    except:
        reader = csv.reader([line.replace("\n",'')])
        csvresult=next(reader)
    return csvresult

def robust_csv_reader(csv_file_path):
    """
    case like 
    data/archive_tex.colorful.addbbl.perfect_successed/1003/1003.5280/boxed_pdf_image/text_bbox/page_0.csv
    has x0 in data, which is not good for parsing
    """
    if csv_file_path.endswith('.json'):
        df = pd.read_json(csv_file_path)
        df.columns = ["markdown", "pdf", 'status', 'text_type', 'bbox']
        return df
    file_path = csv_file_path

    # Initialize a list to hold the cleaned rows
    cleaned_rows = []

    # Define the expected columns
    expected_columns = ["indice", "markdown", "pdf", "status", "text_type", "bbox"]

    # Read the CSV file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        whole_rows = []
        buffer = ""
        next_number = 0
        for line in file:
            parts = line.split(',')
            if len(parts)>0 and parts[0] == str(next_number):
                ### now it is new line and we
                whole_rows.append(buffer.rstrip())
                next_number+=1
                buffer=line
            else:
                buffer += line
        if buffer:
            whole_rows.append(buffer.rstrip())
    bbox_pattern = re.compile(r'\"\[\[.*?\]\]\"')
    whole_rows=whole_rows[1:]

    cleaned_rows = []
    for row in whole_rows:
        # Find bbox match
        parts = parse_csv_line(row)
        if len(parts) == 6:
            cleaned_rows.append(parts)
            continue
        bbox_match = bbox_pattern.search(row)
        if bbox_match:
            bbox = bbox_match.group(0).strip('"')
            row = bbox_pattern.sub('', row, 1)  # Remove the first occurrence of bbox from the row
        else:
            bbox = None

        # Parse the remaining row parts
        parts = parse_csv_line(row)
        if len(parts) >= 5:
            cleaned_row = parts[:5] + [bbox]

            if len(cleaned_row) == len(expected_columns):
                cleaned_rows.append(cleaned_row)
            else:
                print(f"Malformed row (wrong number of columns): {cleaned_row}")
                raise
        else:
            print(f"Malformed row (too few columns): {parts}")
            raise
    df = pd.DataFrame(cleaned_rows, columns=expected_columns)
    df = df[["markdown", "pdf", "status", "text_type", "bbox"]].copy()
    return df
