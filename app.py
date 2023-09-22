from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from werkzeug.utils import secure_filename
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.output_parsers import RegexParser
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import logging
import boto3
from botocore.exceptions import NoCredentialsError
from langchain.document_loaders import UnstructuredURLLoader


logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

logger = logging.getLogger('HELLO WORLD')

UPLOAD_FOLDER = 'D:/ai/backend-python/backend/docs'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

S3_BUCKET_NAME = 'chat-qna'
S3_REGION = 'us-east-1'

s3_re = boto3.resource(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('access_key') ,
    aws_secret_access_key=os.getenv('secret_key') 
)
bucket = s3_re.Bucket('chat-qna')
s3 = boto3.client(
    service_name='s3',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('access_key') ,
    aws_secret_access_key=os.getenv('secret_key') 
)


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('api_key') 

# loader = DirectoryLoader(f'docs', glob="./*.pdf", loader_cls=PyPDFLoader)
chat_history = []

pdf_list = []

# @app.route("/get_pdf", methods=["POST"])
# def get_selected_pdf():
#     file = request.files['selected_pdf']
#     print(file)
#     filename = secure_filename(file.filename)
#     pdf = [f"""https://chat-qna.s3.amazonaws.com/{filename.replace(" ", "+").replace('"', '%22').replace(':', '%3A').replace(',', '%2C')}"""]
#     return pdf
prompt_template = """Given the following extracted parts of a long document and a question, create a final answer in multiple paragraphs. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    Context:
    ---------
    {context}
    ---------
    Question: {question}
    Helpful Answer:"""


output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
    output_parser=output_parser
)
memory = ConversationBufferMemory(
    output_key='answer',
    memory_key='chat_history', return_messages=True)

vectorestore = []
@app.route('/send_pdf_names', methods=['POST'])
def receive_pdf_names():
    data = request.get_json()
    pdf_list.clear()  # Clear the array to remove any previous elements
    pdf_list.extend(data.get('pdfFileNames', []))
    # pdf_list=pdf_file_names
    # Process the pdf_file_names array as needed
    for i in range(len(pdf_list)):
        pdf_list[i] = f"""https://chat-qna.s3.amazonaws.com/{pdf_list[i].replace(" ", "+").replace('"', '%22').replace(':', '%3A').replace(',', '%2C')}"""

    print("SELECTED PDFS:", pdf_list)

    # links = []
    # for obj in bucket.objects.all():
    #     links.append(f"""https://chat-qna.s3.amazonaws.com/{str(obj.key).replace(" ", "+").replace('"', '%22').replace(':', '%3A').replace(',', '%2C')}""")
    links = pdf_list
    print("Links",links)
    loaders = UnstructuredURLLoader(links)
    documents = loaders.load()


    # documents = loader.load()
    chunk_size_value = 2500
    chunk_overlap=0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_value, chunk_overlap=chunk_overlap,length_function=len)
    texts = text_splitter.split_documents(documents)
    docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())
    print("DOC EMBEDDINGS:",docembeddings)
    vectorestore.extend([docembeddings])


    return jsonify({"Status":"Now you can query on the pdfs"})

def load_chain(docembeddings):
    # chain = load_qa_chain(OpenAI(temperature=0.6), chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=800, temperature = 0.4),
        retriever=docembeddings.as_retriever(search_type = "similarity", search_kwargs={"k":3}),
        memory=memory,
        chain_type="stuff", 
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    return conversation_chain

def getanswer2(query):
    conversation_chain = load_chain(vectorestore[-1])
    res = conversation_chain({'question': query, 'chat_history': chat_history})
    chat_history.extend([(query, res["answer"])])
    output= {"Answer":res["answer"]}
    print("CHAT HISTORY:", chat_history)
    return output

# def getanswer(query):
#     relevant_chunks = docembeddings.similarity_search_with_score(query)
#     chunk_docs=[]
#     for chunk in relevant_chunks:
#         chunk_docs.append(chunk[0])
#     results = chain({"input_documents": chunk_docs, "question": query})
#     text_reference=""
#     for i in range(len(results["input_documents"])):
#         text_reference+=results["input_documents"][i].page_content
#     output={"Answer":results["output_text"],"Reference":text_reference}
#     return output



@app.route('/docqna',methods = ["POST"])
def processclaim():
    try:
        input_json = request.get_json(force=True)
        query = input_json["query"]
        print(type(query))
        output=getanswer2(query)
        return output
    except Exception as e:
        return jsonify({"Status":f"{e}"})
    



# @app.route('/upload', methods=['POST'])
# def fileUpload():
#     target=os.path.join(UPLOAD_FOLDER)
#     if not os.path.isdir(target):
#         os.mkdir(target)
#     logger.info("welcome to upload`")
#     file = request.files['file'] 
#     filename = secure_filename(file.filename)
#     destination="/".join([target, filename])
#     file.save(destination)
#     session['uploadFilePath']=destination
#     response="Whatever you wish too return"
#     return response    


def upload_file_to_s3(file, acl="public-read"):
    filename = secure_filename(file.filename)
    print("filename",filename)
    try:
        s3.upload_fileobj(
            file,
            os.getenv("AWS_BUCKET_NAME"),
            filename,
            ExtraArgs={
                "ACL": acl,
                "ContentType": file.content_type
            }
        )

    except Exception as e:
        # This is a catch all exception, edit this part to fit your needs.
        print("Something Happened: ", e)
        return e
    

    # after upload file to s3 bucket, return filename of the uploaded file
    return file.filename

@app.route("/upload", methods=["POST"])
def create():

    # after confirm 'user_file' exist, get the file from input
    file = request.files['file']

    # check whether the file extension is allowed (eg. png,jpeg,jpg,gif)
    if file:
        output = upload_file_to_s3(file) 

        if output:
            return jsonify({'message': f'{output}'})
        
@app.route("/selected_pdfs", methods=["POST"])
def selected_pdfs():
    links = []
    for obj in bucket.objects.all():
        links.append(f"""{str(obj.key).replace(" ", "+").replace('"', '%22').replace(':', '%3A').replace(',', '%2C')}""")
    return jsonify({"pdfs":links})

@app.route("/delete_pdf", methods=["POST"])
def delete_object_from_bucket():
    file = request.get_json()
    print(file)
    file = file['delete_file']
    file_name = file
    bucket_name = 'chat-qna'
    session = boto3.Session(
        aws_access_key_id=os.getenv('access_key') ,
        aws_secret_access_key=os.getenv('secret_key') 
    )
    s3 = session.resource('s3')
    response = s3.meta.client.delete_object(Bucket=bucket_name, Key=file_name)
    # aws_access_key_id=os.getenv('access_key') ,
    # aws_secret_access_key=os.getenv('secret_key') 
    # # s3_client = boto3.client("s3", region_name='us-east-1', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
    # # response = s3_client.delete_object(Bucket=bucket_name, Key=file_name)
    return jsonify({"Status":f"{response}"})



# @app.route('/upload', methods=['POST'])
# def upload_file():
#     try:
#         print("step1")
#         file = request.files['file']
#         print("step2")
#         if file:
#             print("step3")
#             # s3 = boto3.client('s3')
#             s3 = boto3.resource('s3')
#             print("step4")
#             path = "/uploads/"
#             print("step5")
#             s3.Object(S3_BUCKET_NAME, path + file.filename).put(Body=file)

#             # print("step4")
#             # bucket = s3.Bucket(S3_BUCKET_NAME)
#             # print("step5")
#             # bucket.upload_fileobj(file, secure_filename(file.filename))
#             # s3.upload_fileobj(file, S3_BUCKET_NAME, secure_filename(file.filename))
#             return jsonify({'message': 'File uploaded successfully'})
#         else:
#             return jsonify({'error': 'No file part'})

#     except Exception as e:
#         return jsonify({'error':f'{e}'})

    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8095, debug=True)


