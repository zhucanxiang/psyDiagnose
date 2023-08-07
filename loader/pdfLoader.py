import os
import re
import fitz
from typing import List, Dict
from paddleocr import PaddleOCR
from utils.utils import normalize_string
from loader.baseLoader import BaseLoader

file_name_config = {
    "ICD-11精神科（中文版）": "《ICD-11精神科（中文版）》"
}

class ICD11Loader(BaseLoader):
    def __init__(
        self,
        file_path: str,
        extract_image: bool = False
    ):
        self.file_name = file_path.split("/")[-1].split(".")[0]
        self.extract_image = extract_image
        self.seps = {"\t", "\n"}
        if extract_image:
            self.ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, show_log=False)
            tmp_dir = r"./tmp"
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            self.tmp_img_path = os.path.join(tmp_dir, self.file_name + "_tmp_img.png")
        
        self.document = fitz.open(file_path)
        # 优先对6[ABCDE][a-zA-Z0-9]{2,3}\.[a-zA-Z0-9]{1,3}进行匹配，未命中进行6[ABCDE][a-zA-Z0-9]{2,3}匹配
        # self.pattern_filter = re.compile(r"(\()(6[ABCDE][a-zA-Z0-9]{2,3}\.[a-zA-Z0-9]{1,3}|6[ABCDE][a-zA-Z0-9]{2,3})(\))")
        self.pattern = re.compile(r"6[ABCDE][a-zA-Z0-9]{2,3}\.[a-zA-Z0-9]{1,3}|6[ABCDE][a-zA-Z0-9]{2,3}")
        self.contents = self.bulid()
    
    def generate_total_description_str(
        self,
        text: str,
        index_pair_list: List
    ) -> List[str]:
        result = []
        for i in range(len(index_pair_list)):
            code = text[index_pair_list[i][0]:index_pair_list[i][1]]
            if i < len(index_pair_list) - 1:
                context = text[index_pair_list[i][1]+1:index_pair_list[i+1][0]]
            else:#=
                context = text[index_pair_list[i][1]+1:]
            
            if context.endswith("（"):
                context = context[:-1]
            if len(context.strip()) > 0:
                result.append(
                    "编码为%s解释为%s" % (normalize_string(code, seps=self.seps), normalize_string(context, seps=self.seps)))
        
        return result
        
    def generate_total_description_json(
        self,
        text: str,
        index_pair_list: List
    ) -> List[Dict]:
        result = []
        for i in range(len(index_pair_list)):
            code = text[index_pair_list[i][0]:index_pair_list[i][1]]
            if i < len(index_pair_list) - 1:
                context = text[index_pair_list[i][1]+1:index_pair_list[i+1][0]]
            else:#=
                context = text[index_pair_list[i][1]+1:]
            
            if context.endswith("（"):
                context = context[:-1]
            if len(context.strip()) > 0:
                result.append({
                    "code": normalize_string(code, seps=self.seps),
                    "context": normalize_string(context, seps=self.seps)
                })
        
        return result

    def bulid(self) -> List:
        contents = list()
        for _, page in enumerate(self.document.pages()):
            # 提取文字
            text = page.get_text()
            all_result = re.finditer(pattern=self.pattern, string=text)
            index_pair_list = []
            for result in all_result:
                index_pair = list(result.span())
                # 判断是否是合法的匹配, (xx)不合法
                if index_pair[0] > 0 and index_pair[1] < len(text) - 1 and \
                    text[index_pair[0] - 1] != "(" and text[index_pair[1]] != ")":
                    index_pair_list.append(index_pair)
            
            contents.extend(self.generate_total_description_str(
                text=text,
                index_pair_list=index_pair_list
            ))
            # 提取图像
            if self.extract_image:
                img_list = page.get_images()
                for img in img_list:
                    pix = fitz.Pixmap(self.document, img[0])
                    if pix.n - pix.alpha >= 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    pix.save(self.tmp_img_path)
                    result = self.ocr.ocr(self.tmp_img_path)
                    ocr_result = [i[1][0] for line in result for i in line]
                    # print("Image:%s" % ocr_result)
                    contents.append(ocr_result)

        return contents[1:]


if __name__=="__main__":
    instance = ICD11Loader(file_path=r"./data/ICD-11精神科（中文版）.pdf", extract_image=False)
    for content in instance.contents:
        print(content)
    print(len(instance.contents))
    print(file_name_config[instance.file_name])
