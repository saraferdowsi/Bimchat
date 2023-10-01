from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import CSVLoader
import openai
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from flask_cors import CORS

load_dotenv()
app = Flask(__name__)

CORS(app)

class ChatBot:

    def __init__(self):
        # Initialize your chatbot components here
        self.raw_text = self.get_csv_text()
        self.text_chunks = self.get_text_chunks(self.raw_text)
        self.vectorstore = self.get_vectorstore(self.text_chunks)
        self.conversation = self.get_conversation_chain(self.vectorstore)

    def get_csv_text(self):
        text = ""
        loader = CSVLoader('./content/dataset.csv', encoding='utf-8')
        for row in loader.load():
            page_content = row.page_content  # Access the page_content attribute
            text += page_content + "\n"
           # Clean up the temporary file
        return text


    def get_text_chunks(self ,text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(self ,text_chunks):
        embeddings = OpenAIEmbeddings()

        if not text_chunks:
            return None  # Handle empty text_chunks

        try:
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            return vectorstore
        except Exception as e:
            print(f"An error occurred while creating the vectorstore: {e}")
            return None

    def get_conversation_chain(self ,vectorstore):
        if vectorstore is None:
            return None  # Handle case when vectorstore is None

        llm = ChatOpenAI()

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain


    def handle_userinput(self, user_question):
        # try:()
        conversation_chain = self.get_conversation_chain(self.vectorstore)
        # Split the dataset text into questions and responses
        lines = self.raw_text.strip().split('\n')
        questions = [line.split(': ')[1] for line in lines if line.startswith('سوال:')]
        similarities = []
        best_match_idx = None
        # Use a pre-trained BERT model for embedding text
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        # Check if the user's question is in the dataset
        if user_question in questions:
            response = conversation_chain({'question': user_question})
            chat_history_conversation = response['chat_history']
            chat_history = []
            for i, message in enumerate(chat_history_conversation):
                if i % 2 == 0:
                    chat_history.append({"role": "user", "content": message.content})
                else:
                    chat_history.append({"role": "assistant", "content": message.content})
            return {"answer_type": "dataset", "chat_history": chat_history}
        else:
          # Encode user_question and compute its embedding
            user_question_tokens = tokenizer(user_question, return_tensors='pt', padding=True, truncation=True)
            user_question_embedding = model(**user_question_tokens).last_hidden_state.mean(dim=1)
            for i, dataset_question in enumerate(questions):
                dataset_question_tokens = tokenizer(dataset_question, return_tensors='pt', padding=True, truncation=True)
                dataset_question_embedding = model(**dataset_question_tokens).last_hidden_state.mean(dim=1)
                similarity = cosine_similarity(user_question_embedding.detach().numpy(), dataset_question_embedding.detach().numpy())
                similarities.append(similarity[0][0])
                # Track the index of the best match
                if best_match_idx is None or similarity > similarities[best_match_idx]:
                    best_match_idx = i   
            threshold = 0.985
            # If no similar question is found in the dataset, ask GPT-3
            if best_match_idx is None or (similarities[best_match_idx] < threshold):  # You can set a threshold
                gpt_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_question}
                    ]
                )
                assistant_response = gpt_response['choices'][0]['message']['content']
                chat_history = []
                chat_history = [{"role": "user", "content": user_question}, {"role": "assistant", "content": assistant_response}]
                return {"answer_type": "gpt", "chat_history": chat_history}
            else: 
                # Fetch the response for the best-matched question
                best_matched_question = questions[best_match_idx]
                response = conversation_chain({'question': best_matched_question})
                chat_history_conversation = response['chat_history']
                chat_history = []
                for i, message in enumerate(chat_history_conversation):
                    if i % 2 == 0:
                        chat_history.append({"role": "similar-question", "content": message.content})
                    else:
                        chat_history.append({"role": "assistant", "content": message.content})

                return {"answer_type": "conceptually_similar", "chat_history": chat_history}
        # except Exception as e:
        #     return {"error": str(e)}
        

chatbot = ChatBot()

@app.route('/ask', methods=['POST']) 
def ask_question():
    if request.headers['Content-Type'] == 'application/json':
        data = request.json
        user_question = data.get("user_question")
        if user_question:
            response_data =chatbot.handle_userinput(user_question)
            return jsonify(response_data)
    else:
        return "Unsupported Media Type", 415
        
if __name__ == '__main__':
    app.run()

