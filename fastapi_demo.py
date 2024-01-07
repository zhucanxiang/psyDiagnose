from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from chains.local_doc_qa import LocalDocQA
from configs.model_config import *
import nltk
import os
import uvicorn


nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path
dsm_v_vs_path = "/root/zhucanxiang/code/psyDiagnose/knowledge_base/dsm-5/vector_store"
embedding_model_dict_list = list(embedding_model_dict.keys())
llm_model_dict_list = list(llm_model_dict.keys())
local_doc_qa = LocalDocQA()
#flag_csv_logger = gr.CSVLogger()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # 挂载静态文件，指定目录
templates = Jinja2Templates(directory="templates")  # 模板目录


def get_answer(query, history, streaming: bool = STREAMING):
    for resp, history in local_doc_qa.get_knowledge_based_answer(
            query=query, vs_path=dsm_v_vs_path, chat_history=history, streaming=streaming):
        source = "\n\n"
        source += "".join(
            [f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]}</summary>\n"""
             f"""{doc.page_content}\n"""
             f"""</details>"""
             for i, doc in
             enumerate(resp["source_documents"])])
        history[-1][-1] += source
        yield history, ""
    logger.info(f"flagging: username={FLAG_USER_NAME},query={query},,history={history}")


@app.get("/home/{username}")
def home(request: Request):
    new_user, chat_history_str = "", ""
    return templates.TemplateResponse("diagnose.html",
                                      {"request": request, "chat_history": chat_history_str})


class PredictData(BaseModel):
    query: str  # 输入


@app.post("/predict")
def predict(request: PredictData):
    response, history = get_answer(request.query, "")
    return_data = {'response': response, "chat_history": history}
    return return_data


if __name__ == "__main__":
    uvicorn.run(app, host="*", port=8083)
