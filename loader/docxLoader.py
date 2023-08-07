import re
from docx import Document
from typing import List, Dict
from utils.utils import normalize_string
from loader.baseLoader import BaseLoader

file_name_config = {
    "CCMD-3": "《中国精神障碍分类与诊断标准第3版(CCMD-3)》"
}

class CCMD3Loader(BaseLoader):
    def __init__(
        self,
        file_path: str
    ):
        self.document = Document(file_path)
        self.contents = self.bulid()
        self.file_name = file_path.split("/")[-1].split(".")[0]

    def generate_total_description_str(
        self,
        big_type: str,
        sub_type: str,
        diagnostic_criteria: str
    ) -> str:
        return "%s%s被分类为%s" % \
            (normalize_string(sub_type), normalize_string(diagnostic_criteria), normalize_string(big_type))
    
    def generate_total_description_json(
        self,
        big_type: str,
        sub_type: str,
        diagnostic_criteria: str
    ) -> Dict:
        return {
            "description": normalize_string(sub_type) + normalize_string(diagnostic_criteria),
            "type": normalize_string(big_type)
        }
    
    def bulid(self) -> List:
        result = list()
        for table in self.document.tables:
            # 遍历每个table
            row = 0
            while row < len(table.rows):
                if len(table.rows[row].cells) == 2 and table.rows[row].cells[0].text.startswith("编码："):
                    sub_type = table.rows[row].cells[1].text
                    big_type = table.rows[row+1].cells[1].text
                    diagnostic_criteria = table.rows[row+2].cells[1].text

                    # 存在诊断标准
                    if len(diagnostic_criteria.strip()) > 0:
                        result.append(self.generate_total_description_str(
                            big_type=big_type,
                            sub_type=sub_type,
                            diagnostic_criteria=diagnostic_criteria
                        ))
                    row += 3
                else:
                    row += 1
        
        return result


if __name__=="__main__":
    instance = CCMD3Loader(file_path=r"./data/CCMD-3.docx")
    for content in instance.contents:
        print(content)
    print(file_name_config[instance.file_name])
