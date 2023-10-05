import pinecone
import openai

#try OpenAI for embedding 
#dimensions 1536 for ChatGPT
#from OpenAI use model:
MODEL = "text-embedding-ada-002"

# Our model is intented to be used as a sentence and short paragraph encoder. 
# Given an input text, it ouptuts a vector which captures the semantic information. 
# The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

openai.api_key="Your OpenAI API key here"

#model = OpenAI(temperature=0, model="gpt-4", streaming=True)
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

pinecone.init(api_key="347b48f6-0a06-4e0f-843a-d23c00067da1", environment="us-west1-gcp-free") 
index = pinecone.Index("demo-index")


def addData(corpusData,url):
   # id = id = index.describe_index_stats()['total_vector_count']
    
    #return existing Pinecone vector count
    id = index.describe_index_stats()['total_vector_count']
    for i in range(len(corpusData)):
        chunk=corpusData[i]
        chunkInfo=(str(id+i),
         get_embedding(chunk),
                 {'title':url,'context':chunk})
                 #store embedding for each chunk using OpenAI
                         
        #insert data as a context using model, create embedding using OpenAI model
        index.upsert(vectors=[chunkInfo])

#after data is indexed, we can start sending queries to Pinecone        
#k how many relevant documents(chunks) do you want to include in your search
def find_match(query,k):
  
    query_em = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']
    result = index.query(query_em, top_k=k, includeMetadata=True)
   
    return [result['matches'][i]['metadata']['title'] for i in range(k)],[result['matches'][i]['metadata']['context'] for i in range(k)]
